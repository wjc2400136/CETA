import torch

from .bert_attack import BertAttack, tokenize_with_offset
from .pgd import PGD


def _release_embedding(output, modality):
    embedding = output.get(f"{modality}_embed")
    if embedding is None:
        return output[f"{modality}_feat"]
    return embedding[:, 0, :] if embedding.ndim == 3 else embedding


class CoAttackPGD(PGD):
    """Image baseline initialization used by the Co-Attack release."""

    def get_image_feat(self, images):
        output = self.model.inference_image(self.normalize(images))
        return _release_embedding(output, "image")

    def init_delta(self, images):
        delta = images.new_zeros(images.shape)
        if self.random_init:
            if self.norm_type == "linfty":
                delta.uniform_(0.0, self.epsilon)
            else:
                delta.uniform_(0.0, 1.0)
                delta = delta - images
                flat = delta.flatten(1)
                norm = flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
                factor = torch.minimum(
                    self.epsilon / (norm + 1e-20),
                    torch.ones_like(norm),
                )
                delta = delta * factor
        return delta.detach().requires_grad_(True)


class CoAttackBertAttack(BertAttack):
    """BERT substitution scored in the Co-Attack release feature space."""

    def _tokenize(self, text):
        # Corrected benchmark revision: skip the leading CLS logit in the
        # unsliced MLM tensor. Historical zero-offset results retain their
        # original identity; this is an explicit candidate-supply repair.
        return tokenize_with_offset(self.tokenizer, text, 1)

    def get_text_feat(self, texts):
        with torch.no_grad():
            text_input = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            output = self.model.inference_text(text_input)
        return _release_embedding(output, "text")


class SepAttack:
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
        batch_size=32,
    ):
        if step_size is None:
            step_size = epsilon / steps * 1.25
        self.model_name = model_name
        self.model = model
        self.device = next(self.model.parameters()).device
        self.ref_model = ref_model.to(self.device)
        self.tokenizer = tokenizer
        # ImageAttacker
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.norm_type = norm_type
        self.random_init = random_init
        self.image_attacker = CoAttackPGD(
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
        # TextAttacker
        self.max_length = max_length
        self.threshold = threshold
        self.num_perturb = num_perturb
        self.topk = topk
        self.batch_size = batch_size
        self.text_attacker = CoAttackBertAttack(
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

    def attack(self, images, texts, *args):
        adv_images, _ = self.image_attacker(images, texts)
        _, adv_texts = self.text_attacker(images, texts)
        return adv_images, adv_texts
