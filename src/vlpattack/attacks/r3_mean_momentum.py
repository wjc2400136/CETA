from __future__ import annotations


import torch


ARM = "R3_MEAN_MOMENTUM_MU1"


DECAY = 1.0


NORMALIZATION_EPSILON = 1e-12


DECAY_ARMS = {
    1.0: ARM,
    0.75: "R3_MEAN_MOMENTUM_MU075",
    0.5: "R3_MEAN_MOMENTUM_MU050",
}


def decay_arm(decay):
    """The completed coefficient and the two explicitly requested tuning values."""
    if type(decay) is not float or decay not in DECAY_ARMS:
        raise ValueError("Declared momentum decay must be 1.0, 0.75 or 0.5")
    return DECAY_ARMS[decay]


def accumulate(aggregate, previous, *, decay=DECAY):
    """FP32 normalization then mu*previous+normalized, with a zero guard."""
    decay_arm(decay)
    if (
        aggregate.shape != previous.shape
        or aggregate.dtype != previous.dtype
        or aggregate.dtype != torch.float32
        or aggregate.device != previous.device
        or aggregate.requires_grad
        or previous.requires_grad
        or not bool(torch.isfinite(aggregate).all())
        or not bool(torch.isfinite(previous).all())
    ):
        raise ValueError("Detached finite matching FP32 optimizer fields required")
    mean_abs = aggregate.abs().mean()
    denominator = mean_abs + NORMALIZATION_EPSILON
    normalized = aggregate / denominator
    momentum = previous + normalized if decay == 1.0 else previous * decay + normalized
    if not bool(torch.isfinite(momentum).all()):
        raise ValueError("Invalid momentum; no repair")
    return momentum, mean_abs, denominator
