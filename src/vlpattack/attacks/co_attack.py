import copy

import torch
import torch.nn.functional as F

from .bert_attack import (
    filter_characters,
    filter_words,
    get_substitues,
)
from .sep_attack import CoAttackBertAttack, CoAttackPGD, SepAttack


def _release_pair_batch(images, text_group_lens):
    if len(text_group_lens) != len(images):
        raise ValueError("text_group_lens must contain one count per image")
    if any(count <= 0 for count in text_group_lens):
        raise ValueError("Co-Attack requires at least one caption per image")
    repeats = torch.as_tensor(text_group_lens, device=images.device, dtype=torch.long)
    paired_images = torch.repeat_interleave(images, repeats, dim=0)
    last_pair_indices = repeats.cumsum(dim=0) - 1
    return paired_images, last_pair_indices


class CoAttack(SepAttack):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=2 / 255,
        steps=10,
        step_size=None,
        norm_type="linfty",
        random_init=True,
        max_length=30,
        threshold=0.3,
        num_perturb=1,
        topk=10,
        batch_size=8,
    ):
        if step_size is None:
            step_size = epsilon / steps * 1.25
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
        self.image_attacker = CoAttackImageAttacker(
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
        self.text_attacker = CoAttackTextAttacker(
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
        if sum(text_group_lens) != len(texts):
            raise ValueError("text_group_lens must map every caption to exactly one image")
        # adv text
        with torch.no_grad():
            adv_texts = self.text_attacker(texts)
            adv_texts_feat = self.get_text_feat(adv_texts)

        # The release attacks every image-caption pair independently. Its retrieval
        # cache then overwrites duplicate image ids in annotation order, leaving the
        # final pair's image. Reproduce that behavior without a second loss variant.
        paired_images, last_pair_indices = _release_pair_batch(images, text_group_lens)
        paired_adv_images = self.image_attacker(paired_images, adv_texts_feat)
        adv_images = paired_adv_images[last_pair_indices]

        return adv_images, adv_texts

    def get_image_feat(self, images):
        return self.image_attacker.get_image_feat(images)

    def get_text_feat(self, texts):
        return self.text_attacker.get_text_feat(texts)


class CoAttackImageAttacker(CoAttackPGD):
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=2 / 255,
        steps=10,
        step_size=None,
        norm_type="linfty",
        random_init=True,
        *args,
        **kwargs,
    ):
        if step_size is None:
            step_size = epsilon / steps * 1.25
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
        self.alpha = 3.0

    def attack(self, images, adv_texts_feat):
        if len(images) != len(adv_texts_feat):
            raise ValueError("Co-Attack requires one adversarial text per paired image")
        with torch.no_grad():
            orig_images_feat = self.get_image_feat(images).detach()

        delta = self.init_delta(images)
        for _ in range(self.steps):
            adv_images_feat = self.get_image_feat(self.transform(images + delta))
            loss_image = self.get_loss_image(adv_images_feat, orig_images_feat)
            loss_image_text = self.get_loss_image_text(adv_images_feat, adv_texts_feat)
            loss = loss_image + self.alpha * loss_image_text
            grad = self.get_grad(loss, delta)
            delta = self.update_delta(delta, images, grad)

        return (images + delta).detach()

    def get_loss_image(self, adv_images_feat, orig_images_feat):
        return super().get_loss(adv_images_feat, orig_images_feat)

    def get_loss_image_text(self, adv_images_feat, adv_texts_feat):
        if adv_images_feat.shape != adv_texts_feat.shape:
            raise ValueError("Co-Attack paired image/text features must have equal shapes")
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        return criterion(
            F.normalize(adv_images_feat, dim=-1).log_softmax(dim=-1),
            F.normalize(adv_texts_feat, dim=-1).softmax(dim=-1),
        )


class CoAttackTextAttacker(CoAttackBertAttack):
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

    def attack(self, texts):
        texts_input = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        # (batch_size, max_length, vocab_size)
        mlm_logits = self.ref_model(
            texts_input.input_ids, attention_mask=texts_input.attention_mask
        ).logits
        # (batch_size, max_length, topk)
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        orig_feat = self.get_text_feat(texts)

        adv_texts = []
        for i, text in enumerate(texts):
            important_scores = self.get_important_scores(text, orig_feat[i])
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, _, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)

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

                substitutes = get_substitues(
                    substitutes,
                    self.tokenizer,
                    self.ref_model,
                    True,
                    word_pred_scores,
                    self.threshold,
                )

                replace_texts = [" ".join(final_words)]
                available_substitutes = [target_word]
                for substitute in substitutes:
                    if substitute == target_word:
                        continue
                    if any(character in substitute for character in filter_characters):
                        continue
                    if substitute in filter_words:
                        continue

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(" ".join(temp_replace))

                replace_feat = self.get_text_feat(replace_texts)
                loss = self.get_loss(replace_feat, orig_feat[i])
                candidate_idx = loss.argmax()

                final_words[top_index] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != target_word:
                    change += 1

            adv_texts.append(" ".join(final_words))

        return adv_texts
