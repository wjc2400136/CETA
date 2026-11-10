import copy

import numpy as np
import torch
from torchvision import transforms

from .bert_attack import (
    BertAttack,
    filter_characters,
    filter_words,
    get_substitues,
    tokenize_with_offset,
)
from .co_attack import CoAttack
from .pgd import PGD
from .text_decision_trace import (
    append_decision_trace,
    begin_caption_trace,
    finish_caption_trace,
)


class SGA(CoAttack):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=2 / 255,
        steps=10,
        step_size=0.5 / 255,
        norm_type="linfty",
        random_init=True,
        max_length=30,
        threshold=0.3,
        num_perturb=1,
        topk=10,
        batch_size=8,
    ):
        super().__init__(
            model_name,
            model,
            ref_model,
            tokenizer,
            epsilon,
            steps,
            step_size,
            norm_type,
            random_init,
            max_length,
            threshold,
            num_perturb,
            topk,
            batch_size,
        )
        self.image_attacker = SGAImageAttacker(
            self.model_name,
            self.model,
            self.ref_model,
            self.tokenizer,
            self.epsilon,
            self.steps,
            self.step_size,
            self.norm_type,
            self.random_init,
        )
        self.text_attacker = SGATextAttacker(
            self.model_name,
            self.model,
            self.ref_model,
            self.tokenizer,
            self.max_length,
            self.threshold,
            self.num_perturb,
            self.topk,
            self.batch_size,
        )
        self._text_decision_trace_enabled = False
        self._text_decision_trace = []

    def enable_text_decision_trace(self):
        self._text_decision_trace_enabled = True
        self._text_decision_trace = []

    def drain_text_decision_trace(self):
        rows = self._text_decision_trace
        self._text_decision_trace = []
        return rows

    def attack(self, images, texts, text_group_lens):
        images = images.clone().detach().to(self.device)

        # text -> image guided -> adv text
        with torch.no_grad():
            images_feat = self.get_image_feat(images)
            adv_texts = self.text_attacker(texts, images_feat, text_group_lens)
            adv_texts_feat = self.get_text_feat(adv_texts)

        # image -> adv text guided -> adv image
        adv_images = self.image_attacker(images, adv_texts_feat, text_group_lens)

        # text -> adv image guided -> adv text
        with torch.no_grad():
            adv_images_feat = self.get_image_feat(adv_images)
            adv_texts = self.text_attacker(
                texts,
                adv_images_feat,
                text_group_lens,
                trace_sink=(
                    self._text_decision_trace if self._text_decision_trace_enabled else None
                ),
                trace_pass="second_final_text_pass",
            )

        return adv_images, adv_texts


class SGAImageAttacker(PGD):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=2 / 255,
        steps=10,
        step_size=0.5 / 255,
        norm_type="linfty",
        random_init=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name,
            model,
            ref_model,
            tokenizer,
            epsilon,
            steps,
            step_size,
            norm_type,
            random_init,
            *args,
            **kwargs,
        )
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    def attack(self, images, texts_feat, text_group_lens):
        delta = self.init_delta(images)

        for _ in range(self.steps):
            total_grad = 0.0
            for scale in self.scales:
                adv_images_feat = self.get_image_feat(self.transform(images + delta, scale))
                loss = self.get_loss(adv_images_feat, texts_feat, text_group_lens)
                grad = self.get_grad(loss, delta)
                total_grad += grad
            delta = self.update_delta(delta, images, total_grad)

        return (images + delta).detach()

    def transform(self, image, scale):
        if scale == 1.0:
            return image
        ori_shape = (image.shape[-2], image.shape[-1])
        reverse = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC)
        scale_shape = (int(scale * ori_shape[0]), int(scale * ori_shape[1]))
        scale_transform = transforms.Resize(
            scale_shape, interpolation=transforms.InterpolationMode.BICUBIC
        )
        scaled_image = image + torch.from_numpy(
            np.random.normal(0.0, 0.05, image.shape)
        ).float().to(self.device)
        scaled_image = scale_transform(scaled_image)
        scaled_image = torch.clamp(scaled_image, 0.0, 1.0)

        reversed_image = reverse(scaled_image)

        return reversed_image

    def get_loss(self, images_feat, texts_feat, text_group_lens):
        B = images_feat.shape[0]

        it_sim_matrix = images_feat @ texts_feat.T
        it_labels = torch.zeros_like(it_sim_matrix, device=self.device)

        text_start = 0
        for image_idx in range(B):
            text_end = text_start + text_group_lens[image_idx]
            it_labels[image_idx, text_start:text_end] = 1.0
            text_start = text_end

        loss = -(it_sim_matrix * it_labels).sum(-1).mean()

        return loss


class SGATextAttacker(BertAttack):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        max_length=30,
        threshold=0.3,
        num_perturb=1,
        topk=10,
        batch_size=32,
    ):
        super().__init__(
            model_name,
            model,
            ref_model,
            tokenizer,
            max_length,
            threshold,
            num_perturb,
            topk,
            batch_size,
        )

    def _tokenize(self, text):
        # The corrected SGA/DRA/SA-AET benchmark revision aligns word spans
        # with the unsliced MLM tensor after its leading CLS prediction.
        return tokenize_with_offset(self.tokenizer, text, 1)

    def get_important_scores(self, text, orig_feat):
        # Author word identification uses text-only KL. Candidate selection
        # still uses this class's image-guided negative-dot-product get_loss.
        masked_texts = [" ".join(words) for words in self._get_masked(text)]
        masked_feats = []
        for start in range(0, len(masked_texts), self.batch_size):
            masked_feats.append(self.get_text_feat(masked_texts[start : start + self.batch_size]))
        masked_feats = torch.cat(masked_feats, dim=0)
        return BertAttack.get_loss(self, masked_feats, orig_feat)

    def attack(
        self,
        texts,
        images_feat,
        text_group_lens,
        *,
        trace_sink=None,
        trace_pass="final_text_pass",
    ):
        texts_input = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        mlm_logits = self.ref_model(
            texts_input.input_ids, attention_mask=texts_input.attention_mask
        ).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        orig_feat = self.get_text_feat(texts)

        text2image_idx = []
        for image_idx, text_num in enumerate(text_group_lens):
            text2image_idx += [image_idx] * text_num

        adv_texts = []
        for i, text in enumerate(texts):
            image_idx = text2image_idx[i]
            important_scores = self.get_important_scores(text, orig_feat[i])
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, _, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            caption_trace = (
                begin_caption_trace(
                    caption_index=i,
                    source_text=text,
                    words=words,
                    wordpiece_keys=keys,
                    pass_label=trace_pass,
                )
                if trace_sink is not None
                else None
            )

            change = 0
            for top_index, _ in list_of_index:
                if change >= self.num_perturb:
                    break

                target_word = words[top_index]
                if target_word in filter_words:
                    continue
                if keys[top_index][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index][0] : keys[top_index][1]]
                word_pred_scores = word_pred_scores_all[i, keys[top_index][0] : keys[top_index][1]]
                mlm_id_slice = substitutes
                mlm_score_slice = word_pred_scores

                substitute_metadata = [] if caption_trace is not None else None
                substitutes = get_substitues(
                    substitutes,
                    self.tokenizer,
                    self.ref_model,
                    True,
                    word_pred_scores,
                    self.threshold,
                    metadata_sink=substitute_metadata,
                )

                replace_texts = [" ".join(final_words)]
                available_substitutes = [target_word]
                available_metadata = (
                    [
                        {
                            "kind": "no_change_sentinel",
                            "token_ids": [],
                            "wordpieces": [],
                            "decoded_string": target_word,
                        }
                    ]
                    if caption_trace is not None
                    else None
                )
                for substitute_index, substitute in enumerate(substitutes):
                    if substitute == target_word:
                        continue
                    if any(character in substitute for character in filter_characters):
                        continue
                    if substitute in filter_words:
                        continue

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index] = substitute
                    available_substitutes.append(substitute)
                    if available_metadata is not None and substitute_metadata is not None:
                        available_metadata.append(substitute_metadata[substitute_index])
                    replace_texts.append(" ".join(temp_replace))

                replace_feat = self.get_text_feat(replace_texts)
                loss = self.get_loss(replace_feat, images_feat[image_idx])
                candidate_idx = loss.argmax()

                final_words[top_index] = available_substitutes[candidate_idx]

                change_increment = int(available_substitutes[candidate_idx] != target_word)
                change += change_increment
                if caption_trace is not None:
                    selected_metadata = available_metadata[int(candidate_idx.item())]
                    append_decision_trace(
                        caption_trace,
                        word_position=top_index,
                        target_raw=target_word,
                        wordpiece_key_span=keys[top_index],
                        raw_candidates=available_substitutes,
                        selected_candidate_index=int(candidate_idx.item()),
                        mlm_id_slice=mlm_id_slice,
                        mlm_score_slice=mlm_score_slice,
                        change_increment=change_increment,
                        change_cumulative=change,
                        selected_candidate_token_ids=selected_metadata["token_ids"],
                        selected_candidate_wordpieces=selected_metadata["wordpieces"],
                        selected_candidate_metadata_kind=selected_metadata["kind"],
                    )

            adv_texts.append(" ".join(final_words))
            if caption_trace is not None:
                trace_sink.append(finish_caption_trace(caption_trace, final_words=final_words))

        return adv_texts

    def get_loss(self, text_feat, image_feat):
        loss = -text_feat @ image_feat.unsqueeze(-1)
        loss = loss.squeeze(-1)
        return loss
