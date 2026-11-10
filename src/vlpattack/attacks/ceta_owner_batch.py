"""CETA original-owner batching, isolated endpoint profile fix 13975.

The seven current B1 files remain byte-for-byte authoritative for B1. This
separate adapter changes only the owner axis of the image optimization. It
does not supply text, choose candidates, select a recipe, load models, save
images, or launch an experiment. Numerical functions are SERVER ONLY.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any


PORT_VERSION = "ceta-native-original-owner-batch-endpoint-microbatch1-SOURCE-13975-v2"
TRACE_VERSION = "CETA_ORIGINAL_OWNER_BATCH_STEP_TRACE_ENDPOINT_MICROBATCH1_13975_V2"
MAX_ORIGINAL_OWNERS = 4
CURRENT_B1_CORE_PIN = (31619, '020fe31c168ab6488f4643b832a2629c8cb510a82fd64fe6143414cd3b1f7311')
CURRENT_B1_NATIVE_INPUTS_PIN = (17003, '1cd79e44cb7e0444f1919a3acb339fb581bf0f6c14c365c0004e26b7dbc99fa9')


def _need(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def _require_server() -> None:
    if sys.platform != "linux" or os.environ.get("VLPATTACK_SERVER_EXECUTION") != "1":
        raise RuntimeError("CETA owner-batch numerical execution is SERVER ONLY")


def _check_module(module: Any, expected: tuple[int, str]) -> None:
    """Check the actual bound Source bytes; runtime-origin acceptance is external."""
    raw = Path(module.__file__).resolve(strict=True).read_bytes()
    _need((len(raw), hashlib.sha256(raw).hexdigest()) == expected,
          "The current exact B1 Source identity changed")


def _checked_core(core: Any) -> Any:
    _require_server()
    _check_module(core, CURRENT_B1_CORE_PIN)
    _need(core.CORE_VERSION == "csga-server-core-SOURCE-only-20261025-9276-v1",
          "Use the current13900 B1 core, never a rewritten B1 oracle")
    return core


class DirectNativeOwnerBatchEndpoint:
    """N logical original owners through the exact original N1 image endpoint.

    The complete native DirectNativeImageEndpoint performs its own original
    pixel normalization, inference and unit checks on each graph-connected N1
    slice. Only the resulting units are concatenated. Physical forward events
    and counters describe those actual N1 calls; the optimizer remains N-owner.
    Backend, precision, guard tolerance and model training state are unchanged.
    """

    def __init__(self, source: str, model: Any, *, native_inputs: Any,
                 count: Any = None):
        _require_server()
        _check_module(native_inputs, CURRENT_B1_NATIVE_INPUTS_PIN)
        native_inputs.source_spec(source)
        _need(getattr(model, "training", None) is False, "Qualified eval source required")
        _need(count is None or callable(count), "Optional actual operation sink must be callable")
        self.source, self.model, self.native_inputs, self.count = source, model, native_inputs, count
        self._ceta_interpolated_view_guard_version = native_inputs.INTERPOLATED_VIEW_GUARD_VERSION
        self.forward_attempts = 0
        self.forward_completions = 0
        self.forward_input_batch_sizes: list[int] = []
        self.logical_forward_attempts = 0
        self.logical_forward_completions = 0
        self.logical_forward_input_batch_sizes: list[int] = []
        self._active_original_forward_metadata: dict[str, Any] | None = None
        # Reuse the actual pinned B1 endpoint, including normalization; do not
        # reimplement a model-only chunk after normalizing the full N tensor.
        self._original_endpoint = native_inputs.DirectNativeImageEndpoint(
            source, model, count=self._count_original_forward_attempt)

    def _count_original_forward_attempt(self, event: Any) -> None:
        """Observe the original endpoint's actual pre-inference event once."""
        _need(type(event) is dict and event.get("kind") == "source_image_forward"
              and event.get("source") == self.source and event.get("inputs") == 1
              and type(self._active_original_forward_metadata) is dict,
              "Original N1 endpoint forward event required")
        self.forward_attempts += 1
        self.forward_input_batch_sizes.append(event["inputs"])
        if self.count is not None:
            self.count(dict(event, **self._active_original_forward_metadata,
                original_owner_batch=True, model_owner_microbatch=1,
                whole_original_N1_endpoint_reused=True,
                physical_forward_attempt=self.forward_attempts))

    def _ceta_interpolated_view_call(self, pixels: Any) -> Any:
        return self(pixels, interpolated_view=True)

    def __call__(self, pixels: Any, *, interpolated_view: bool = False) -> Any:
        _require_server()
        import torch

        _need(type(interpolated_view) is bool, "Explicit interpolated-view guard mode required")
        native, spec = self.native_inputs, self.native_inputs.source_spec(self.source)
        _need(isinstance(pixels, torch.Tensor) and pixels.dtype == torch.float32
              and pixels.ndim == 4 and 1 <= int(pixels.shape[0]) <= MAX_ORIGINAL_OWNERS
              and tuple(pixels.shape[1:]) == (3, spec["side"], spec["side"]),
              "N original native RGB carriers required; no resize/cast/chunk fallback")
        n = int(pixels.shape[0])
        tolerance = native.INTERPOLATED_VIEW_PIXEL_TOLERANCE if interpolated_view else 0.0
        finite_mask = torch.isfinite(pixels)
        if not bool(finite_mask.all()) or not bool(((pixels >= -tolerance) & (pixels <= 1.0 + tolerance)).all()):
            observation = native._native_pixel_guard_observation(
                pixels, finite_mask, tolerance=tolerance, interpolated_view=interpolated_view)
            raise ValueError("Finite valid native owner-batch pixels required; actual_guard_observation=" + repr(observation))
        self.logical_forward_attempts += 1
        self.logical_forward_input_batch_sizes.append(n)
        units = []
        try:
            for index in range(n):
                self._active_original_forward_metadata = dict(
                    logical_N_owner_endpoint_call=self.logical_forward_attempts,
                    logical_original_owner_batch_size=n, owner_local_index=index,
                    physical_call_owner_count=1,
                    event_reference_is_shared_ledger_not_an_N_image_model_call=True)
                # A slice retains the shared N-leaf graph. All normalization and
                # inference operators below belong to the unchanged B1 endpoint.
                unit = self._original_endpoint(
                    pixels[index:index + 1], interpolated_view=interpolated_view)
                self.forward_completions += 1
                units.append(unit)
            result = units[0] if n == 1 else torch.cat(units, dim=0)
            self.logical_forward_completions += 1
            return result
        finally:
            self._active_original_forward_metadata = None


def _owner_side_view(point: Any, index: int, draw: Any, owner: Any, *, core: Any) -> Any:
    """Current B1 resize/noise body applied to a graph-connected N-leaf slice.

    A slice is deliberately not a new leaf. Detaching it would destroy the
    shared N-leaf graph. All original value checks and operators are retained.
    """
    import math
    import numpy as np
    import torch
    from torchvision import transforms

    height = owner.carrier_size
    _need(type(draw) is core.ScaleDraw and draw.slot in (0, 2), "An original side draw is required")
    side = "down" if draw.slot == 0 else "up"
    _need(type(draw.noise_seed) is int and isinstance(draw.noise, np.ndarray)
          and draw.noise.dtype == np.float32 and draw.noise.shape == (1, 3, height, height)
          and bool(np.isfinite(draw.noise).all()), "Original owner-local source-grid FP32 noise required")
    if draw.mode == "no_scale":
        valid = draw.rho == 0.0 and draw.scale == 1.0 and draw.scale_seed is None
    else:
        low, high = (1.0 - draw.rho, 1.0) if side == "down" else (1.0, 1.0 + draw.rho)
        valid = (draw.mode == "interval" and draw.rho in (0.25, 0.5, 0.75, 1.0)
                 and low <= draw.scale < high and type(draw.scale_seed) is int)
    _need(valid and np.isfinite(draw.scale), "Original scale sample required; no repair")
    size = max(1, math.floor(draw.scale * height))
    _need(draw.native_size == (size, size), "Original discrete native side size differs")
    current = point[index:index + 1].to("cpu") + torch.from_numpy(draw.noise)
    first = transforms.Resize(draw.native_size, interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=False)(current).clamp(0.0, 1.0)
    return transforms.Resize((height, height), interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=False)(first).to(point.device)


def _readonly_draws(draws_by_owner: Any) -> Any:
    """Observer-only immutable byte-backed noise copies; no tape replacement."""
    from dataclasses import replace
    import numpy as np

    result = []
    for draws in draws_by_owner:
        copied = []
        for draw in draws:
            noise = None if draw.noise is None else np.frombuffer(
                draw.noise.tobytes(order="C"), dtype=np.float32).reshape(draw.noise.shape)
            copied.append(replace(draw, noise=noise))
        result.append(tuple(copied))
    return tuple(result)


def run_image_path_owners(clean: Any, selections: Any, endpoint: Any, *, core: Any,
                          on_step: Any = None, on_draw: Any = None,
                          on_state: Any = None) -> tuple[Any, list[dict[str, Any]]]:
    """Ten synchronous updates of ONE N-owner pixel leaf, N in 1..4.

    For each original down/direct/up slot, the logical endpoint receives all N
    images and reuses N original N1 endpoints internally. Independent FP64 dots
    are summed only
    to obtain their block-separable VJP with respect to the N leaf; branches
    remain separate VJPs. Fields are then averaged in the exact original
    down/direct/up FP32 order. Canonical accumulate and NativeBox are called
    on each owner separately before the whole current tensor is replaced.
    """
    core = _checked_core(core)
    import numpy as np
    import torch
    from vlpattack.attacks import paired_endpoint, r3_mean_momentum

    for module, relative in ((paired_endpoint, "src/vlpattack/attacks/paired_endpoint.py"),
                             (r3_mean_momentum, "src/vlpattack/attacks/r3_mean_momentum.py")):
        raw = Path(module.__file__).resolve(strict=True).read_bytes()
        _need(hashlib.sha256(raw).hexdigest() == core.CANONICAL_RUNTIME_DEPENDENCIES[relative],
              "Original canonical projection/momentum Source changed")
    _need(type(selections) in (tuple, list) and 1 <= len(selections) <= MAX_ORIGINAL_OWNERS,
          "One to four complete original-owner selections required")
    n = len(selections)
    _need(all(type(s) is core.OwnerSelection and type(s.prepared) is core.PreparedFull for s in selections),
          "Use immutable selections from the current original B1 preparation")
    owners = tuple(core._checked_owner(s.prepared.owner) for s in selections)
    recipes = tuple(core._checked_recipe(s.recipe) for s in selections)
    source, height, dimension = owners[0].source, owners[0].carrier_size, owners[0].dimension
    _need(len({o.owner for o in owners}) == n and len({o.owner_count for o in owners}) == 1
          and len({o.owner_group_manifest_sha256 for o in owners}) == 1
          and len({o.input_binding_sha256 for o in owners}) == 1
          and all(o.source == source and o.carrier_size == height and o.dimension == dimension for o in owners),
          "Distinct global owners from one native source/domain required")
    _need(all(r == recipes[0] and r.control == "full" and r.path_mode == "three_view" for r in recipes),
          "One unchanged Full recipe per owner-batch required")
    _need(isinstance(clean, torch.Tensor) and clean.dtype == torch.float32
          and tuple(clean.shape) == (n, 3, height, height) and clean.is_leaf and not clean.requires_grad
          and bool(torch.isfinite(clean).all()) and not bool((clean < 0).any()) and not bool((clean > 1).any()),
          "One detached N-owner native FP32 carrier leaf in [0,1] required")
    _need(callable(endpoint) and (on_step is None or callable(on_step))
          and (on_draw is None or callable(on_draw)) and (on_state is None or callable(on_state)),
          "Explicit native batch endpoint/observers required")
    guides = []
    for selection in selections:
        guide = selection.guide_cpu
        _need(isinstance(guide, np.ndarray) and guide.dtype == np.float64
              and guide.shape == (dimension,) and bool(np.isfinite(guide).all()) and not guide.flags.writeable,
              "Frozen complete-owner CPU FP64 guide required")
        guides.append(torch.from_numpy(guide.copy()).to(clean.device))
    guide = torch.stack(guides)
    _need(guide.dtype == torch.float64 and not guide.requires_grad,
          "Do not cast or renormalize the original independent guides")
    # NativeBox keeps its original FP64 legal box / FP32 inward-nextafter rules.
    boxes = tuple(paired_endpoint.NativeBox(clean[i:i + 1]) for i in range(n))
    streams = tuple(core.ScaleStreams(o, r) for o, r in zip(owners, recipes, strict=True))
    current = torch.cat([box.clean.clone() for box in boxes], dim=0)
    momentum = torch.zeros_like(clean)
    traces: list[dict[str, Any]] = []
    counters = dict(endpoint_attempts=0, endpoint_completions=0, VJP_attempts=0,
                    VJP_completions=0, synchronous_projection_barriers=0)
    endpoint_owner = getattr(endpoint, "__self__", endpoint)
    physical_endpoint = endpoint_owner if isinstance(endpoint_owner, DirectNativeOwnerBatchEndpoint) else None
    stage, active_step = "not_started", None
    try:
        for step in range(core.STEPS):
            active_step, stage = step + 1, "original_owner_draws"
            point = current.detach().requires_grad_(True)
            _need(point.is_leaf and tuple(point.shape) == (n, 3, height, height),
                  "The optimizable leaf must retain the real original-owner batch axis")
            draws_by_owner = tuple(stream.draw((1, 3, height, height)) for stream in streams)
            _need(all(tuple(d.slot for d in draws) == (0, 1, 2) for draws in draws_by_owner),
                  "Original down/direct/up branch order required")
            if on_draw is not None:
                on_draw(step + 1, tuple(o.owner for o in owners), _readonly_draws(draws_by_owner))
            physical_attempt_start = None if physical_endpoint is None else len(physical_endpoint.forward_input_batch_sizes)
            physical_completion_start = None if physical_endpoint is None else physical_endpoint.forward_completions
            fields, units, losses, view_hashes = [], [], [], []
            for branch, slot in enumerate((0, 1, 2)):
                stage = "branch_view"
                if slot == 1:
                    _need(all(d[branch].scale == 1.0 and d[branch].native_size == (height, height)
                              and d[branch].noise is None and d[branch].scale_seed is None
                              and d[branch].noise_seed is None for d in draws_by_owner),
                          "Direct branch must be the original N leaf without transform or RNG")
                    transformed = point
                else:
                    transformed = torch.cat([_owner_side_view(point, i, draws_by_owner[i][branch],
                                                              owners[i], core=core) for i in range(n)], dim=0)
                if on_step is not None:
                    view_hashes.append([hashlib.sha256(transformed[i:i + 1].detach().cpu().numpy().tobytes()).hexdigest()
                                        for i in range(n)])
                stage = "native_N_owner_forward"
                counters["endpoint_attempts"] += 1
                unit = endpoint(transformed)
                counters["endpoint_completions"] += 1
                _need(tuple(unit.shape) == (n, dimension), "The endpoint changed the true owner batch size")
                for i, owner in enumerate(owners):
                    core._checked_endpoint_unit(unit[i:i + 1], owner, point.device, differentiable=True)
                # Independent dot products; NO owner mean, negatives or shared normalization.
                owner_losses = torch.stack([(unit[i].double() * guide[i]).sum() for i in range(n)])
                stage = "original_branch_N_leaf_VJP"
                counters["VJP_attempts"] += 1
                field = torch.autograd.grad(owner_losses.sum(), point)[0]
                counters["VJP_completions"] += 1
                _need(field.shape == point.shape and field.dtype == torch.float32
                      and bool(torch.isfinite(field).all()), "Finite FP32 fields for every original owner required")
                fields.append(field.detach())
                units.append(unit.detach())
                losses.append(owner_losses.detach())
            stage = "ordered_branch_FP32_mean"
            aggregate = torch.zeros_like(point)
            for field in fields:
                aggregate += field
            aggregate /= len(fields)
            incoming_momentum = momentum
            next_momentum, means, denominators = [], [], []
            for i in range(n):
                value, mean_abs, denominator = r3_mean_momentum.accumulate(
                    aggregate[i:i + 1], momentum[i:i + 1], decay=1.0)
                next_momentum.append(value)
                means.append(mean_abs)
                denominators.append(denominator)
            momentum = torch.cat(next_momentum, dim=0)
            stage = "synchronous_original_native_projection"
            proposals64 = tuple(point[i:i + 1].detach().double()
                                - core.STEP_SIZE * momentum[i:i + 1].sign().double() for i in range(n))
            after = torch.cat([box.native(proposal) for box, proposal in zip(boxes, proposals64, strict=True)], dim=0)
            counters["synchronous_projection_barriers"] += 1
            if on_state is not None:
                stage = "readonly_actual_state_observer"
                on_state(step + 1, dict(point=point, aggregate=aggregate,
                    incoming_momentum=incoming_momentum, mean_abs=tuple(means),
                    denominator=tuple(denominators), momentum=momentum,
                    proposals64=proposals64, projection_lower64=tuple(b.lower for b in boxes),
                    projection_upper64=tuple(b.upper for b in boxes), after=after,
                    normalized_gradient_observation="Read the original accumulate frame local 'normalized'; never reverse momentum"))
            owner_traces = []
            for i, (owner, recipe, draws) in enumerate(zip(owners, recipes, draws_by_owner, strict=True)):
                trace = dict(step=step + 1, core_version=core.CORE_VERSION, owner=owner.owner,
                    carrier_size=owner.carrier_size, recipe=core.recipe_metadata(recipe),
                    slots=[d.slot for d in draws], scales=[d.scale for d in draws],
                    native_sizes=[list(d.native_size) for d in draws],
                    scale_seeds=[d.scale_seed for d in draws], noise_seeds=[d.noise_seed for d in draws],
                    noise_roles=[None if d.slot == 1 else "noise:down" if d.slot == 0 else "noise:up" for d in draws],
                    sigmas=[0.0 if d.slot == 1 else recipe.sigma for d in draws],
                    weights=[1 / len(draws)] * len(draws), field_count=len(fields),
                    noise_position="before_first_resize", interpolation="bilinear", antialias=False,
                    clamp_position="after_first_resize_before_return_resize",
                    scores=[float(value[i]) for value in losses], mean_abs=float(means[i]),
                    normalization_denominator=float(denominators[i]),
                    normalization_epsilon=core.NORMALIZATION_EPSILON, decay=recipe.mu,
                    step_size=core.STEP_SIZE, seed_namespace=core.SEED_NAMESPACE,
                    guide_reselected=False, new_discrete_size_rule="max(1,floor(scale*carrier_size))")
                if on_step is not None:
                    trace["view_sha256"] = [row[i] for row in view_hashes]
                owner_traces.append(trace)
            batch_trace = dict(schema=TRACE_VERSION, port_version=PORT_VERSION, step=step + 1,
                owner_ids=[o.owner for o in owners], owner_traces=owner_traces,
                noise_sha256_by_owner=[[None if d.noise is None else hashlib.sha256(d.noise.tobytes(order="C")).hexdigest()
                                        for d in draws] for draws in draws_by_owner],
                batch_operator=dict(optimization_leaf_shape=list(point.shape),
                    logical_owner_endpoint_input_batch_sizes=[n] * len(fields),
                    native_forward_input_batch_sizes=None if physical_endpoint is None else list(
                        physical_endpoint.forward_input_batch_sizes[physical_attempt_start:]),
                    completed_source_model_endpoint_calls=None if physical_endpoint is None else
                        physical_endpoint.forward_completions - physical_completion_start,
                    physical_model_calls_observed_at_original_N1_endpoint=physical_endpoint is not None,
                    model_owner_microbatch_size=None if physical_endpoint is None else 1,
                    branch_slots=[0, 1, 2],
                    completed_native_endpoint_calls=len(fields), completed_original_N_leaf_VJP_calls=len(fields),
                    owner_field_participations=n * len(fields), synchronized_owner_update_barrier=True,
                    original_owner_updates=n, forward_microbatching_by_this_port=None if physical_endpoint is None else True,
                    counters_through_step=dict(counters),
                    endpoint_calls_are_not_a_claim_about_unobserved_internal_model_kernels=True))
            if on_step is not None:
                stage = "readonly_step_observer"
                on_step(batch_trace, current, torch.stack(fields), momentum, after, torch.stack(units))
            traces.append(batch_trace)
            # The entire N-owner state advances only after all fields/projections.
            current = after
        return current, traces
    except BaseException as error:
        try:
            error.ceta_owner_batch_failure_observation = dict(port_version=PORT_VERSION,
                owner_ids=[o.owner for o in owners], optimization_tensor_shape=list(clean.shape),
                active_step=active_step, stage=stage, actual_counters=dict(counters),
                completed_trace_steps=len(traces), automatic_retry=False)
        except Exception:
            pass
        raise
