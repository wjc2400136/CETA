from __future__ import annotations

import copy

import torch

from .bert_attack import (
    BertAttack,
    filter_characters,
    filter_words,
    get_substitues,
)
from .pgd import PGD
from .text_decision_trace import (
    append_decision_trace,
    begin_caption_trace,
    finish_caption_trace,
)


def _text_to_image(text_group_lens: list[int] | tuple[int, ...]) -> list[int]:
    mapping: list[int] = []
    for image_index, count in enumerate(text_group_lens):
        mapping.extend([image_index] * count)
    return mapping


class SimVLATextAttacker(BertAttack):
    """Cross-modal word identification and image-guided substitution."""

    def _cross_modal_scores(self, text: str, image_feature: torch.Tensor) -> torch.Tensor:
        masked_texts = [" ".join(words) for words in self._get_masked(text)]
        features = []
        for start in range(0, len(masked_texts), self.batch_size):
            features.append(self.get_text_feat(masked_texts[start : start + self.batch_size]))
        masked = torch.cat(features, dim=0)
        criterion = torch.nn.KLDivLoss(reduction="none")
        target = image_feature.softmax(dim=-1).expand(len(masked), -1)
        return criterion(masked.log_softmax(dim=-1), target).sum(dim=-1)

    def attack(
        self,
        texts,
        image_features,
        text_group_lens,
        *,
        trace_sink=None,
        trace_pass="final_text_pass",
    ):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.ref_model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
        scores, predictions = torch.topk(logits, self.topk, dim=-1)
        mapping = _text_to_image(text_group_lens)
        if len(mapping) != len(texts):
            raise ValueError("Caption groups do not cover every SimVLA text")

        adversarial: list[str] = []
        for text_index, text in enumerate(texts):
            feature_index = text_index if len(image_features) == len(texts) else mapping[text_index]
            image_feature = image_features[feature_index]
            importance = self._cross_modal_scores(text, image_feature)
            positions = sorted(enumerate(importance), key=lambda item: item[1], reverse=True)
            words, _, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            caption_trace = (
                begin_caption_trace(
                    caption_index=text_index,
                    source_text=text,
                    words=words,
                    wordpiece_keys=keys,
                    pass_label=trace_pass,
                )
                if trace_sink is not None
                else None
            )
            changes = 0
            for position, _ in positions:
                if changes >= self.num_perturb:
                    break
                target = words[position]
                if target in filter_words or keys[position][0] > self.max_length - 2:
                    continue
                mlm_id_slice = predictions[text_index, keys[position][0] : keys[position][1]]
                mlm_score_slice = scores[text_index, keys[position][0] : keys[position][1]]
                candidate_metadata = [] if caption_trace is not None else None
                candidates = get_substitues(
                    mlm_id_slice,
                    self.tokenizer,
                    self.ref_model,
                    True,
                    mlm_score_slice,
                    self.threshold,
                    metadata_sink=candidate_metadata,
                )
                replacement_texts = [" ".join(final_words)]
                replacements = [target]
                replacement_metadata = (
                    [
                        {
                            "kind": "no_change_sentinel",
                            "token_ids": [],
                            "wordpieces": [],
                            "decoded_string": target,
                        }
                    ]
                    if caption_trace is not None
                    else None
                )
                for candidate_index, candidate in enumerate(candidates):
                    if (
                        candidate == target
                        or candidate in filter_words
                        or any(character in candidate for character in filter_characters)
                    ):
                        continue
                    proposal = copy.deepcopy(final_words)
                    proposal[position] = candidate
                    replacements.append(candidate)
                    if replacement_metadata is not None and candidate_metadata is not None:
                        replacement_metadata.append(candidate_metadata[candidate_index])
                    replacement_texts.append(" ".join(proposal))
                replacement_features = self.get_text_feat(replacement_texts)
                losses = -(replacement_features @ image_feature.unsqueeze(-1)).squeeze(-1)
                selected = int(losses.argmax().item())
                final_words[position] = replacements[selected]
                change_increment = int(replacements[selected] != target)
                changes += change_increment
                if caption_trace is not None:
                    selected_metadata = replacement_metadata[selected]
                    append_decision_trace(
                        caption_trace,
                        word_position=position,
                        target_raw=target,
                        wordpiece_key_span=keys[position],
                        raw_candidates=replacements,
                        selected_candidate_index=selected,
                        mlm_id_slice=mlm_id_slice,
                        mlm_score_slice=mlm_score_slice,
                        change_increment=change_increment,
                        change_cumulative=changes,
                        selected_candidate_token_ids=selected_metadata["token_ids"],
                        selected_candidate_wordpieces=selected_metadata["wordpieces"],
                        selected_candidate_metadata_kind=selected_metadata["kind"],
                    )
            adversarial.append(" ".join(final_words))
            if caption_trace is not None:
                trace_sink.append(finish_caption_trace(caption_trace, final_words=final_words))
        return adversarial


class SimVLAImageAttacker(PGD):
    def __init__(
        self,
        *args,
        epsilon=2 / 255,
        steps=10,
        step_size=0.5 / 255,
        num_samples=4,
        noise_factor=16.0,
        reverse_step_factor=0.1,
        **kwargs,
    ):
        super().__init__(*args, epsilon=epsilon, steps=steps, step_size=step_size, **kwargs)
        self.num_samples = num_samples
        self.noise_factor = noise_factor
        self.reverse_step_factor = reverse_step_factor

    def _noisy_views(self, images: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        radius = self.noise_factor * self.epsilon if scale is None else scale
        views = [images]
        for _ in range(self.num_samples):
            noise = torch.empty_like(images).uniform_(-radius, radius)
            views.append(images + noise)
        return torch.stack(views, dim=0)

    def _alignment_loss(self, image_features, text_features, text_group_lens):
        similarity = image_features @ text_features.T
        labels = torch.zeros_like(similarity)
        start = 0
        for image_index, count in enumerate(text_group_lens):
            labels[image_index, start : start + count] = 1.0
            start += count
        return -(similarity * labels).sum(dim=-1).mean()

    def attack(self, images, text_features, text_group_lens):
        images = images.detach().to(self.device)
        adversarial = torch.clamp(
            images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon), 0.0, 1.0
        )
        view_count = self.num_samples + 1
        previous_gradient_delta = torch.zeros(
            (view_count, *images.shape), device=images.device, dtype=images.dtype
        )
        momentum = torch.zeros_like(images)

        for _ in range(self.steps):
            views = self._noisy_views(adversarial)
            current_gradients = torch.zeros_like(previous_gradient_delta)
            total_gradient = torch.zeros_like(images)
            for view_index in range(view_count):
                probe = (
                    (
                        views[view_index]
                        + self.reverse_step_factor
                        * self.epsilon
                        * previous_gradient_delta[view_index]
                    )
                    .detach()
                    .requires_grad_(True)
                )
                features = self.get_image_feat(probe)
                loss = self._alignment_loss(features, text_features, text_group_lens)
                gradient = torch.autograd.grad(loss, probe, retain_graph=False)[0]
                current_gradients[view_index] = gradient
                total_gradient += gradient
            normalized_views = current_gradients / (
                current_gradients.abs().mean(dim=(2, 3, 4), keepdim=True) + 1e-12
            )
            previous_gradient_delta = (normalized_views - previous_gradient_delta).detach()
            gradient = total_gradient / view_count
            gradient = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = momentum + gradient
            adversarial = adversarial.detach() + self.step_size * momentum.sign()
            adversarial = torch.minimum(
                torch.maximum(adversarial, images - self.epsilon), images + self.epsilon
            )
            adversarial = torch.clamp(adversarial, 0.0, 1.0)
        return adversarial.detach()


class SimVLA:
    """Author-release-aligned two-stage SimVLA adapter."""

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
        batch_size=32,
        num_samples=4,
        noise_factor=16.0,
        reverse_step_factor=0.1,
        semantic_topk=1,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.semantic_topk = int(semantic_topk)
        if self.semantic_topk < 0:
            raise ValueError("semantic_topk must be non-negative")
        self.text_attacker = SimVLATextAttacker(
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
        self.image_attacker = SimVLAImageAttacker(
            model_name,
            model,
            ref_model,
            tokenizer,
            epsilon=epsilon,
            steps=steps,
            step_size=step_size,
            norm_type=norm_type,
            random_init=random_init,
            num_samples=num_samples,
            noise_factor=noise_factor,
            reverse_step_factor=reverse_step_factor,
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

    def _abstract_texts(self, texts, mapped_image_features):
        abstracted: list[str] = []
        for index, text in enumerate(texts):
            words = text.split()
            if len(words) <= 1:
                abstracted.append(text)
                continue
            scores = self.text_attacker._cross_modal_scores(text, mapped_image_features[index])
            remove_count = min(self.semantic_topk, len(words) - 1)
            if remove_count == 0:
                abstracted.append(text)
                continue
            removed = set(torch.topk(scores, remove_count, largest=False).indices.tolist())
            abstracted.append(" ".join(word for i, word in enumerate(words) if i not in removed))
        return abstracted

    def attack(self, images, texts, text_group_lens):
        images = images.detach().to(self.device)
        with torch.no_grad():
            views = self.image_attacker._noisy_views(images)
            view_features = []
            for view in views:
                view_features.append(self.image_attacker.get_image_feat(view))
            averaged_images = torch.stack(view_features, dim=0).mean(dim=0)
            adversarial_texts = self.text_attacker.attack(
                texts,
                averaged_images,
                text_group_lens,
                trace_sink=(
                    self._text_decision_trace if self._text_decision_trace_enabled else None
                ),
                trace_pass="final_text_pass",
            )
            mapping = _text_to_image(text_group_lens)
            mapped_images = averaged_images[mapping]
            guidance_texts = self._abstract_texts(adversarial_texts, mapped_images)
            guidance_features = self.text_attacker.get_text_feat(guidance_texts)
        adversarial_images = self.image_attacker.attack(images, guidance_features, text_group_lens)
        return adversarial_images, adversarial_texts
