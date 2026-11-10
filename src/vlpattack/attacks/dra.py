import copy
import random

import torch

from .bert_attack import filter_characters, filter_words, get_substitues
from .sga import SGA, SGAImageAttacker, SGATextAttacker
from .text_decision_trace import (
    append_decision_trace,
    begin_caption_trace,
    finish_caption_trace,
)


class DRA(SGA):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=8 / 255,
        steps=10,
        step_size=2 / 255,
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
        self.image_attacker = DRAImageAttacker(
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
        self.text_attacker = DRATextAttacker(
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

    def attack(self, images, texts, text_group_lens):
        images = images.clone().detach().to(self.device)

        # text -> image guided -> adv text
        with torch.no_grad():
            images_feat = self.get_image_feat(images)
            adv_texts = self.text_attacker(texts, images_feat, text_group_lens)
            adv_texts_feat = self.get_text_feat(adv_texts)

        # image -> adv text guided -> adv image
        adv_images, last_adv_images = self.image_attacker(images, adv_texts_feat, text_group_lens)

        # text -> (image + adv image + last adv image) guided -> adv text
        with torch.no_grad():
            adv_images_feat = self.get_image_feat(adv_images)
            last_adv_images_feat = self.get_image_feat(last_adv_images)
            adv_texts = self.text_attacker(
                texts,
                images_feat,
                text_group_lens,
                adv_images_feat,
                last_adv_images_feat,
                trace_sink=(
                    self._text_decision_trace if self._text_decision_trace_enabled else None
                ),
                trace_pass="second_final_text_pass",
            )

        return adv_images, adv_texts


class DRAImageAttacker(SGAImageAttacker):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=8 / 255,
        steps=10,
        step_size=2 / 255,
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
        self.num_sample = 5

    def attack(self, images, texts_feat, text_group_lens):
        return self._trajectory_attack(
            images,
            lambda image_features: self.get_loss(image_features, texts_feat, text_group_lens),
        )

    def _scaled_grad(self, images, delta, objective):
        total_grad = torch.zeros_like(delta)
        for scale in self.scales:
            image_features = self.get_image_feat(self.transform(images + delta, scale))
            total_grad += self.get_grad(objective(image_features), delta)
        return total_grad

    def _lookahead_candidate(self, current_delta, images, sample_grad):
        del images
        return current_delta.detach() + self.step_size * sample_grad.sign()

    def _trajectory_attack(self, images, objective):
        """Author-release DRA trajectory sampling and update."""
        delta = self.init_delta(images)
        last_delta = delta.detach().clone()

        for step in range(self.steps):
            current_delta = delta.detach().clone()
            if step == 0:
                probe = delta
            else:
                best_score = float("-inf")
                best_ratios = None
                for _ in range(self.num_sample):
                    ratios = self.rand3Num()
                    sample_delta = (
                        ((ratios[0] / 100) * current_delta + (ratios[2] / 100) * last_delta)
                        .detach()
                        .requires_grad_(True)
                    )
                    sample_features = self.get_image_feat(images + sample_delta)
                    sample_grad = self.get_grad(objective(sample_features), sample_delta)
                    candidate = self._lookahead_candidate(current_delta, images, sample_grad)
                    with torch.no_grad():
                        score = float(objective(self.get_image_feat(images + candidate)).item())
                    if score > best_score:
                        best_score = score
                        best_ratios = ratios
                if best_ratios is None:
                    raise RuntimeError("DRA failed to select a trajectory sample")
                probe = (
                    ((best_ratios[0] / 100) * current_delta + (best_ratios[2] / 100) * last_delta)
                    .detach()
                    .requires_grad_(True)
                )

            gradient = self._scaled_grad(images, probe, objective)
            last_delta = current_delta
            delta = self.update_delta(current_delta, images, gradient)

        return (images + delta).detach(), (images + last_delta).detach()

    def rand3Num(self):
        # current adversarial, clean, previous adversarial
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            if 1 <= num3 <= 100:
                return (num1, num2, num3)

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


class DRATextAttacker(SGATextAttacker):
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
        self.text_loss_ratios = [0.6, 0.2, 0.2]

    def attack(
        self,
        texts,
        images_feat,
        text_group_lens,
        adv_images_feat=None,
        last_adv_images_feat=None,
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

                if adv_images_feat is None and last_adv_images_feat is None:
                    loss = self.get_loss(replace_feat, images_feat[image_idx])
                else:
                    loss_0 = self.text_loss_ratios[0] * self.get_loss(
                        replace_feat, images_feat[image_idx]
                    )
                    loss_1 = self.text_loss_ratios[1] * self.get_loss(
                        replace_feat, adv_images_feat[image_idx]
                    )
                    loss_2 = self.text_loss_ratios[2] * self.get_loss(
                        replace_feat, last_adv_images_feat[image_idx]
                    )
                    loss = loss_0 + loss_1 + loss_2

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
