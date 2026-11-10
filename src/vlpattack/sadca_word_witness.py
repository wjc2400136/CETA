"""Observe and validate native original-word slots; never infer edits by alignment.

No tensors, tokenizers, models or RNG are used here. A raw multiword replacement
is one value in the author's original ``words`` list, and thus one source slot.
"""
from __future__ import annotations

import operator

WITNESS_SCHEMA = "SADCA_FINAL_NATIVE_ORIGINAL_WORD_SLOTS_V2"


def _integer(value, label):
    if isinstance(value, bool):
        raise ValueError(label + " cannot be Boolean")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(label + " must be an integer") from error


def source_word_witness(original_text, original_words, final_raw_words, native_change_count):
    """Copy native slots after search; blank replacements are checked on output."""
    original_words, final_raw_words = list(original_words), list(final_raw_words)
    if (not isinstance(original_text, str) or not original_words
            or len(original_words) != len(final_raw_words)
            or any(not isinstance(word, str) for word in original_words + final_raw_words)
            or " ".join(original_words) != original_text):
        raise ValueError("SADCA source-word witness has inconsistent original slots")
    changes = [position for position, (old, raw) in
               enumerate(zip(original_words, final_raw_words, strict=True)) if old != raw]
    native_change_count = _integer(native_change_count, "Native accepted-change count")
    if native_change_count != len(changes):
        raise ValueError("SADCA native change counter disagrees with actual original slots")
    return {"schema": WITNESS_SCHEMA,
            "original_text": original_text, "original_words": original_words,
            "final_raw_words": final_raw_words,
            "adversarial_text": " ".join(final_raw_words),
            "changed_original_positions": changes,
            "raw_replacements": [{"original_position": position,
                                   "original_word": original_words[position],
                                   "replacement_raw": final_raw_words[position]}
                                  for position in changes],
            "native_accepted_change_count": native_change_count}


def validate_source_word_witnesses(original_texts, adversarial_texts, witnesses, budget):
    """Validate the final round only, without replacing or repairing any output."""
    budget = _integer(budget, "SADCA output text budget")
    if budget not in range(1, 6):
        raise ValueError("SADCA output text budget must be from 1 to 5")
    original_texts, adversarial_texts, witnesses = (
        list(original_texts), list(adversarial_texts), list(witnesses))
    if not (len(original_texts) == len(adversarial_texts) == len(witnesses)):
        raise ValueError("SADCA final text/witness cardinalities disagree")
    for original, adversarial, observed in zip(
            original_texts, adversarial_texts, witnesses, strict=True):
        if not isinstance(observed, dict) or observed.get("schema") != WITNESS_SCHEMA:
            raise ValueError("SADCA final original-word witness is missing")
        rebuilt = source_word_witness(
            observed["original_text"], observed["original_words"],
            observed["final_raw_words"], observed["native_accepted_change_count"])
        if observed != rebuilt or original != rebuilt["original_text"] or adversarial != rebuilt["adversarial_text"]:
            raise ValueError("SADCA final original-word witness is inconsistent with output")
        if len(rebuilt["changed_original_positions"]) > budget:
            raise ValueError("SADCA final text exceeds its original-word-slot budget")
        if any(not edit["replacement_raw"].strip() for edit in rebuilt["raw_replacements"]):
            raise ValueError("SADCA final text has a whitespace-only raw replacement")
    return witnesses
