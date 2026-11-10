import random

import torch

from .dra import DRA, DRAImageAttacker, DRATextAttacker


class SA_AET(DRA):
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
        self.sample_texts = 0.4
        self.image_attacker = SA_AETImageAttacker(
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
        self.text_attacker = SA_AETTextAttacker(
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

    def attack(self, images, texts, text_group_lens, projection_matrix):
        images = images.clone().detach().to(self.device)

        # text -> image guided -> adv text
        with torch.no_grad():
            images_feat = self.get_image_feat(images)
            adv_texts = self.text_attacker(texts, images_feat, text_group_lens)
            adv_texts_feat = self.get_text_feat(adv_texts)

        # image -> adv text guided -> adv image
        adv_images, last_adv_images = self.image_attacker(
            images, adv_texts_feat, text_group_lens, projection_matrix
        )

        # text -> (image + adv image + last adv image) guided -> adv text
        with torch.no_grad():
            adv_images_feat = self.get_image_feat(adv_images)
            last_adv_images_feat = self.get_image_feat(last_adv_images)
            adv_texts = self.text_attacker(
                texts, images_feat, text_group_lens, adv_images_feat, last_adv_images_feat
            )

        return adv_images, adv_texts

    def get_projection_matrix(self, all_texts):
        all_texts_feat = []
        total_num = len(all_texts)

        with torch.no_grad():
            for start_idx in range(0, total_num, 256):
                end_idx = min(start_idx + 256, total_num)
                batch_texts = all_texts[start_idx:end_idx]

                text_input = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                text_output = self.model.inference_text(text_input)
                batch_text_feat = text_output["text_feat"]

                all_texts_feat.append(batch_text_feat)

        all_texts_feat = torch.cat(all_texts_feat, dim=0)

        U, _, _ = torch.svd(all_texts_feat.T.to(torch.float32))
        projection_matrix = U[:, 1 : len(U)] @ U[:, 1 : len(U)].t()
        return projection_matrix


class SA_AETImageAttacker(DRAImageAttacker):
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

    def attack(self, images, texts_feat, text_group_lens, projection_matrix):
        return self._trajectory_attack(
            images,
            lambda image_features: self.get_loss(
                image_features, texts_feat, text_group_lens, projection_matrix
            ),
        )

    def get_loss(self, images_feat, texts_feat, text_group_lens, projection_matrix):
        B = images_feat.shape[0]

        images_feat = images_feat @ projection_matrix
        texts_feat = texts_feat @ projection_matrix

        it_sim_matrix = images_feat @ texts_feat.T
        it_labels = torch.zeros_like(it_sim_matrix, device=self.device)

        text_start = 0
        for image_idx in range(B):
            text_end = text_start + text_group_lens[image_idx]
            it_labels[image_idx, text_start:text_end] = 1.0
            text_start = text_end

        loss = -(it_sim_matrix * it_labels).sum(-1).mean()

        return loss

    def _lookahead_candidate(self, current_delta, images, sample_grad):
        return self.update_delta(current_delta, images, sample_grad)

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
            if 1 <= num3 <= 100 and num1 < num3 < num2:
                return (num1, num2, num3)


class SA_AETTextAttacker(DRATextAttacker):
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
