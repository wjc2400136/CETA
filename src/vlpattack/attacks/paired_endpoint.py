from __future__ import annotations


import os


import sys


from pathlib import Path


def require_server():
    if sys.platform == "linux" and os.environ.get("VLPATTACK_SERVER_EXECUTION") == "1":
        return
    raise RuntimeError("Use the documented Linux runtime")


class NativeBox:
    """Shared behavior-neutral legal RGB projection and inward native rounding."""

    def __init__(self, clean):
        require_server()
        import torch

        if (
            not isinstance(clean, torch.Tensor)
            or clean.dtype != torch.float32
            or clean.ndim != 4
            or tuple(clean.shape[:2]) != (1, 3)
            or min(clean.shape[2:]) < 1
            or not bool(torch.isfinite(clean).all())
            or not bool(((clean >= 0) & (clean <= 1)).all())
        ):
            raise ValueError("One finite float32 RGB image in [0,1] required")
        self.clean = clean.detach().clone()
        self.clean64 = self.clean.double()
        self.epsilon = 2.0 / 255.0
        self.lower = (self.clean64 - self.epsilon).clamp(0, 1)
        self.upper = (self.clean64 + self.epsilon).clamp(0, 1)

    def native(self, ideal):
        import torch

        if ideal.shape != self.clean.shape or not bool(torch.isfinite(ideal).all()):
            raise ValueError("Invalid proposal; no repair")
        ideal = torch.maximum(self.lower, torch.minimum(self.upper, ideal.double()))
        rounded = ideal.float()
        outward = (rounded.double() - self.clean64).abs() > (ideal - self.clean64).abs()
        result = torch.where(outward, torch.nextafter(rounded, self.clean), rounded).detach()
        if not bool(((result.double() >= self.lower) & (result.double() <= self.upper)).all()):
            raise RuntimeError("Native rounding violated original budget")
        return result

    def reflect(self, image):
        return self.native(self.lower + self.upper - image.double())

    def step(self, image, field, steps):
        import torch

        if (
            field.shape != self.clean.shape
            or field.dtype != self.clean.dtype
            or field.device != self.clean.device
            or not bool(torch.isfinite(field).all())
        ):
            raise ValueError("Invalid whole-leaf field; no repair")
        return self.native(image.double() - (self.epsilon / steps) * field.sign().double())
