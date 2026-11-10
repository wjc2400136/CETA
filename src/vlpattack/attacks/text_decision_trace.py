"""Behavior-neutral text-decision trace helpers.

The helpers only serialize values already computed by an attack.  They never
invoke a model, consume RNG, filter/reorder candidates, or choose a candidate.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..data.schema import clean_caption

TRACE_SCHEMA_VERSION = 1


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_content_identity(value: torch.Tensor) -> dict[str, object]:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def rng_identity() -> dict[str, object]:
    numpy_state = np.random.get_state()
    identity = {
        "python_sha256": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy_sha256": canonical_sha256(
            {
                "algorithm": numpy_state[0],
                "state_sha256": hashlib.sha256(
                    np.ascontiguousarray(numpy_state[1]).tobytes()
                ).hexdigest(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            }
        ),
        "torch_cpu_sha256": hashlib.sha256(
            torch.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "torch_cuda_device": None,
        "torch_cuda_sha256": None,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        state = torch.cuda.get_rng_state(device)
        identity["torch_cuda_device"] = int(device)
        identity["torch_cuda_sha256"] = hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    return identity


def begin_caption_trace(
    *,
    caption_index: int,
    source_text: str,
    words: Sequence[str],
    wordpiece_keys: Sequence[Sequence[int]],
    pass_label: str,
) -> dict[str, Any]:
    word_list = [str(word) for word in words]
    keys = [[int(value) for value in key] for key in wordpiece_keys]
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "caption_index": int(caption_index),
        "pass": pass_label,
        "source_exact_input": source_text,
        "source_exact_input_sha256": text_sha256(source_text),
        "source_caption_sha256": text_sha256(source_text),
        "source_words": word_list,
        "source_word_list": word_list,
        "source_words_count": len(word_list),
        "source_word_list_sha256": canonical_sha256(word_list),
        "source_words_sha256": canonical_sha256(word_list),
        "source_wordpiece_key_spans": keys,
        "source_wordpiece_key_spans_sha256": canonical_sha256(keys),
        "decisions": [],
        "rng_identity_before": rng_identity(),
    }


def append_decision_trace(
    caption: dict[str, Any],
    *,
    word_position: int,
    target_raw: str,
    wordpiece_key_span: Sequence[int],
    raw_candidates: Sequence[str],
    selected_candidate_index: int,
    mlm_id_slice: torch.Tensor,
    mlm_score_slice: torch.Tensor,
    change_increment: int,
    change_cumulative: int,
    selected_candidate_token_ids: Sequence[int],
    selected_candidate_wordpieces: Sequence[str],
    selected_candidate_metadata_kind: str,
) -> None:
    candidates = [str(value) for value in raw_candidates]
    selected_index = int(selected_candidate_index)
    if not candidates or not 0 <= selected_index < len(candidates):
        raise ValueError("Trace received an invalid already-selected candidate identity")
    selected = candidates[selected_index]
    if not selected:
        raise ValueError("Trace selected candidate must be non-empty")
    decision = {
        "decision_index": len(caption["decisions"]),
        "word_position": int(word_position),
        "target_raw": str(target_raw),
        "wordpiece_key_span": [int(value) for value in wordpiece_key_span],
        "raw_candidates": candidates,
        "raw_candidate_count": len(candidates),
        "raw_candidate_list_sha256": canonical_sha256(candidates),
        "selected_raw_candidate": selected,
        "selected_raw_candidate_sha256": text_sha256(selected),
        "selected_candidate_index": selected_index,
        "selected_candidate_token_ids": [int(value) for value in selected_candidate_token_ids],
        "selected_candidate_wordpieces": [str(value) for value in selected_candidate_wordpieces],
        "selected_candidate_effective_surface_tokens": [
            str(value) for value in clean_caption(selected, 30).split()
        ],
        "selected_candidate_metadata_kind": str(selected_candidate_metadata_kind),
        "mlm_id_slice": tensor_content_identity(mlm_id_slice),
        "mlm_score_slice": tensor_content_identity(mlm_score_slice),
        "change_increment": int(change_increment),
        "change_cumulative": int(change_cumulative),
    }
    caption["decisions"].append(decision)


def finish_caption_trace(
    caption: dict[str, Any],
    *,
    final_words: Sequence[str],
) -> dict[str, Any]:
    source_words = [str(value) for value in caption["source_words"]]
    final = [str(value) for value in final_words]
    source_word_list_length_preserved = len(source_words) == len(final)
    changed = [
        index
        for index, (source, target) in enumerate(zip(source_words, final, strict=True))
        if source != target
    ]
    raw_caption = " ".join(final)
    effective_caption = clean_caption(raw_caption, 30)
    decisions = caption["decisions"]
    attempted_positions = [int(value["word_position"]) for value in decisions]
    committed_decisions = [value for value in decisions if int(value["change_increment"]) == 1]
    selected_decision = committed_decisions[0] if len(committed_decisions) == 1 else None
    caption.update(
        {
            "decision_count": len(caption["decisions"]),
            "committed_changed_positions": changed,
            "committed_changed_position": changed[0] if changed else None,
            "committed_change_count": len(changed),
            "source_word_list_length_preserved": source_word_list_length_preserved,
            "structural_position_budget_observed": (
                source_word_list_length_preserved and len(changed) <= 1
            ),
            "attempted_source_positions_in_order": attempted_positions,
            "committed_source_positions": changed,
            "committed_position_count": len(changed),
            "selected_source_position": (
                selected_decision["word_position"] if selected_decision else None
            ),
            "source_wordpiece_key_span": (
                selected_decision["wordpiece_key_span"] if selected_decision else None
            ),
            "raw_source_word": (selected_decision["target_raw"] if selected_decision else None),
            "selected_candidate_token_ids": (
                selected_decision["selected_candidate_token_ids"] if selected_decision else []
            ),
            "selected_candidate_wordpieces": (
                selected_decision["selected_candidate_wordpieces"] if selected_decision else []
            ),
            "selected_candidate_raw_decoded_string": (
                selected_decision["selected_raw_candidate"] if selected_decision else None
            ),
            "selected_candidate_raw_decoded_sha256": (
                selected_decision["selected_raw_candidate_sha256"] if selected_decision else None
            ),
            "selected_candidate_effective_surface_tokens": (
                selected_decision["selected_candidate_effective_surface_tokens"]
                if selected_decision
                else []
            ),
            "selected_candidate_index": (
                selected_decision["selected_candidate_index"] if selected_decision else None
            ),
            "selected_candidate_metadata_kind": (
                selected_decision["selected_candidate_metadata_kind"] if selected_decision else None
            ),
            "raw_change_increment": (
                selected_decision["change_increment"] if selected_decision else 0
            ),
            "cumulative_raw_position_commits": len(changed),
            "zero_change": not changed,
            "final_words": final,
            "final_source_word_list": final,
            "final_words_count": len(final),
            "final_words_sha256": canonical_sha256(final),
            "final_source_word_list_sha256": canonical_sha256(final),
            "final_raw_caption": raw_caption,
            "final_raw_caption_sha256": text_sha256(raw_caption),
            "final_effective_caption": effective_caption,
            "final_effective_caption_sha256": text_sha256(effective_caption),
            "raw_adversarial_caption": raw_caption,
            "raw_adversarial_caption_sha256": text_sha256(raw_caption),
            "effective_max_words30_caption": effective_caption,
            "effective_max_words30_caption_sha256": text_sha256(effective_caption),
            "rng_identity_after": rng_identity(),
        }
    )
    caption["source_only_control_flow_identity"] = canonical_sha256(
        {
            "pass": caption["pass"],
            "source_caption_sha256": caption["source_caption_sha256"],
            "attempted_source_positions_in_order": attempted_positions,
            "committed_source_positions": changed,
            "decision_candidate_hashes": [
                value["raw_candidate_list_sha256"] for value in decisions
            ],
            "decision_selected_indices": [value["selected_candidate_index"] for value in decisions],
        }
    )
    return caption


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "append_decision_trace",
    "begin_caption_trace",
    "canonical_sha256",
    "finish_caption_trace",
    "rng_identity",
    "tensor_content_identity",
    "text_sha256",
]
