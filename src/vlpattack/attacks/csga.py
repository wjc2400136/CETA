from __future__ import annotations


import hashlib


import json


import math


import os


import sys


import time


from dataclasses import dataclass, replace


from typing import Any


STATUS = "UNADOPTED_SOURCE_ONLY"


CORE_VERSION = "csga-server-core-SOURCE-only-20261025-9276-v1"


STEPS = 10


STEP_SIZE = 0.5 / 255.0


NORMALIZATION_EPSILON = 1e-12


SEED_NAMESPACE = "ALL_RANDOM_SCALE_v1"


UNIT_TOLERANCE = 2e-6


DEVELOPMENT_DEFINITIONS = (("BOX9_A0p50_S8_255", 9, 0.5, 8),)
CONTROL_KINDS = ("full",)

SOURCE_SPECS = {
    "albef": (384, 256, 30, ("input_ids", "attention_mask")),
    "tcl": (384, 256, 30, ("input_ids", "attention_mask")),
    "clip_vit_b16": (224, 512, 77, ("input_ids",)),
    "clip_rn101": (224, 512, 77, ("input_ids",)),
}


CANONICAL_RUNTIME_DEPENDENCIES = {
    "src/vlpattack/attacks/paired_endpoint.py": "13d4b11720e211c3fe782d9610e766611b092c2dffd5f9c86f92ea243130c070",
    "src/vlpattack/attacks/r3_mean_momentum.py": "2e32210135dec3c02273246d1c9bfe3f368ec5eef52ce34c9d217dc4ac2dd563"
}


def _need(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _sha256_string(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _metadata_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _metadata_sha(value: Any) -> str:
    return hashlib.sha256(_metadata_json(value).encode("utf-8")).hexdigest()


def _require_server_runtime() -> None:
    """Location guard only; this does not grant ROOT adoption or model permission."""
    if sys.platform != "linux" or os.environ.get("VLPATTACK_SERVER_EXECUTION") != "1":
        raise RuntimeError("CSGA numerical arithmetic is SERVER-only and unadopted")


@dataclass(frozen=True, slots=True)
class Recipe:
    recipe_id: str
    control: str
    kernel: int
    rho: float
    sigma_numerator_255: int
    mu: float = 1.0
    path_mode: str = "three_view"
    search_mode: str = "full"
    guide_mode: str = "selected_q"

    @property
    def sigma(self) -> float:
        return self.sigma_numerator_255 / 255.0


def resolve_recipe(recipe_id: str, *, control: str = "full") -> Recipe:
    """Resolve one finite candidate or one stated control around that candidate.

    NoBOX/NoScale/NoNoise require fresh initial selection; they may not be
    applied by changing a fixed Full-Q packet. Existing absent components are
    rejected instead of manufacturing duplicate auxiliary conditions.
    """
    _need(type(recipe_id) is str and control in CONTROL_KINDS,
          "An explicit V10 recipe and controlled condition are required")
    definitions = [row for row in DEVELOPMENT_DEFINITIONS if row[0] == recipe_id]
    _need(len(definitions) == 1, "Unknown or nonfinite development recipe")
    name, kernel, rho, sigma_num = definitions[0]
    result = Recipe(name, "full", kernel, rho, sigma_num)
    if control == "no_box":
        _need(kernel != 1, "The selected candidate already has no BOX")
        return replace(result, control=control, kernel=1)
    if control == "no_scale":
        _need(rho != 0.0, "The selected candidate already has no scale")
        return replace(result, control=control, rho=0.0)
    if control == "no_noise":
        _need(sigma_num != 0, "The selected candidate already has no noise")
        return replace(result, control=control, sigma_numerator_255=0)
    if control == "no_momentum":
        return replace(result, control=control, mu=0.0)
    if control == "point":
        return replace(result, control=control, path_mode="point")
    if control == "o_guide":
        return replace(result, control=control, guide_mode="original_o")
    if control == "importance_first":
        return replace(result, control=control, search_mode="importance_first")
    return result


def _checked_recipe(recipe: Recipe) -> Recipe:
    _need(type(recipe) is Recipe, "A resolved immutable Recipe is required")
    _need(recipe == resolve_recipe(recipe.recipe_id, control=recipe.control),
          "Ad hoc or combined controls are not silently admitted")
    return recipe


def recipe_metadata(recipe: Recipe) -> dict[str, Any]:
    recipe = _checked_recipe(recipe)
    return dict(
        status=STATUS, core_version=CORE_VERSION, recipe_id=recipe.recipe_id,
        control=recipe.control, kernel=recipe.kernel, rho=recipe.rho,
        sigma_numerator_255=recipe.sigma_numerator_255, mu=recipe.mu,
        path_mode=recipe.path_mode, search_mode=recipe.search_mode,
        guide_mode=recipe.guide_mode, steps=STEPS,
        step_size_numerator_255=0.5, epsilon_numerator_255=2,
        normalization_epsilon=NORMALIZATION_EPSILON,
        semantic_key=[recipe.kernel, recipe.rho, recipe.sigma_numerator_255,
                      recipe.mu, recipe.path_mode, recipe.search_mode, recipe.guide_mode],
        metadata_identity_is_not_model_acceptance=True,
    )


@dataclass(frozen=True, slots=True)
class OwnerContract:
    source: str
    owner: int
    owner_count: int
    caption_ids: tuple[int, ...]
    expected_caption_ids: tuple[int, ...]
    owner_group_manifest_sha256: str
    input_binding_sha256: str

    @property
    def carrier_size(self) -> int:
        return SOURCE_SPECS[self.source][0]

    @property
    def dimension(self) -> int:
        return SOURCE_SPECS[self.source][1]

    @property
    def native_context(self) -> int:
        return SOURCE_SPECS[self.source][2]

    @property
    def native_fields(self) -> tuple[str, ...]:
        return SOURCE_SPECS[self.source][3]


def make_owner_contract(*, source: str, owner: int, owner_count: int,
                        caption_ids: Any, expected_caption_ids: Any,
                        owner_group_manifest_sha256: str,
                        input_binding_sha256: str) -> OwnerContract:
    """Bind the complete original owner group without slicing, sorting or padding.

    expected_caption_ids must come from the caller's independent frozen owner
    manifest. The core checks equality, not the truth of that external manifest.
    New arbitrary dataset-sized owner ranges are explicit in owner_count; the
    seed formula for each historical integer owner is unchanged.
    """
    _need(source in SOURCE_SPECS, "One declared native source is required")
    _need(_integer(owner_count, minimum=1) and _integer(owner) and owner < owner_count,
          "An explicit valid owner and total owner count are required")
    _need(type(caption_ids) in (tuple, list) and type(expected_caption_ids) in (tuple, list),
          "Explicit complete ordered caption groups are required")
    observed, expected = tuple(caption_ids), tuple(expected_caption_ids)
    _need(bool(observed) and all(_integer(i) for i in observed),
          "Each owner must contain its nonempty original integer caption group")
    _need(len(set(observed)) == len(observed) and observed == expected,
          "Do not omit, duplicate, reorder, take first-five or pad owner captions")
    _need(_sha256_string(owner_group_manifest_sha256) and _sha256_string(input_binding_sha256),
          "Frozen owner-group and input binding pins are required")
    return OwnerContract(source, owner, owner_count, observed, expected,
                         owner_group_manifest_sha256, input_binding_sha256)


def _checked_owner(owner: OwnerContract) -> OwnerContract:
    _need(type(owner) is OwnerContract, "An explicit OwnerContract is required")
    checked = make_owner_contract(
        source=owner.source, owner=owner.owner, owner_count=owner.owner_count,
        caption_ids=owner.caption_ids, expected_caption_ids=owner.expected_caption_ids,
        owner_group_manifest_sha256=owner.owner_group_manifest_sha256,
        input_binding_sha256=owner.input_binding_sha256,
    )
    _need(checked == owner, "Owner contract drift")
    return owner


def owner_metadata(owner: OwnerContract) -> dict[str, Any]:
    owner = _checked_owner(owner)
    return dict(source=owner.source, owner=owner.owner, owner_count=owner.owner_count,
                caption_ids=list(owner.caption_ids), m_i=len(owner.caption_ids),
                carrier_size=owner.carrier_size, native_dimension=owner.dimension,
                native_context=owner.native_context, native_fields=list(owner.native_fields),
                owner_group_manifest_sha256=owner.owner_group_manifest_sha256,
                input_binding_sha256=owner.input_binding_sha256,
                complete_group_checked_against_external_manifest=True)


@dataclass(frozen=True, slots=True)
class PoolRow:
    index: int
    raw_index: int
    raw: str
    native: str
    position: int | None
    target: str | None
    replacement: str | None
    original_position_commits: int
    native_identity: tuple[tuple[str, tuple[int, ...]], ...]
    original_metadata_json: str


@dataclass(frozen=True, slots=True)
class CanonicalPool:
    rows: tuple[PoolRow, ...]
    native_field_names: tuple[str, ...]
    native_context: int
    representatives: tuple[int, ...]
    aliases_json: tuple[str, ...]
    unencodable_json: tuple[str, ...]
    raw_count: int
    domain_sha256: str


def canonical_pool_snapshot(rows: Any, *, native_field_names: Any, native_context: int,
                            representatives: Any, aliases: Any, raw_count: int,
                            unencodable: Any = ()) -> CanonicalPool:
    """Validate and snapshot existing canonical metadata; perform no filtering.

    aliases keep the historical raw_index/representative_index convention.
    Optional exclusions must already be recorded native_context_overflow rows;
    the core never tokenizes, truncates, repairs or invents an exclusion. Only
    CLIP may consume these exclusions, checked when the source is bound below.
    This reproduces the old overflow-only policy, not a new generic filter.
    """
    _need(type(rows) in (list, tuple) and bool(rows), "A complete nonempty canonical pool is required")
    _need(type(native_field_names) in (tuple, list), "Explicit native token field names are required")
    names = tuple(native_field_names)
    field_map = {"input_ids": "native_token_ids", "attention_mask": "native_attention_mask"}
    _need(bool(names) and len(set(names)) == len(names) and all(n in field_map for n in names),
          "All source-native token fields must be present without duplicates")
    _need(native_context in (30, 77) and _integer(raw_count, minimum=1),
          "Explicit native context and raw supply count are required")
    _need(type(representatives) in (list, tuple) and type(aliases) in (list, tuple)
          and type(unencodable) in (list, tuple), "Canonical representative/alias/overflow metadata required")
    reps = tuple(representatives)
    _need(len(reps) == len(rows) and reps[0] == 0
          and all(_integer(r) and r < raw_count for r in reps)
          and list(reps) == sorted(set(reps)), "Stable first representative order and original index0 required")
    snapshots, identities = [], set()
    for index, (row, raw_index) in enumerate(zip(rows, reps, strict=True)):
        _need(type(row) is dict and row.get("index") == index and type(row.get("index")) is int
              and row.get("raw_index") == raw_index and type(row.get("raw_index")) is int,
              "Preserve exact canonical index and raw representative index")
        _need(type(row.get("raw")) is str and bool(row["raw"].strip())
              and type(row.get("native")) is str and bool(row["native"].strip()),
              "Literal raw and canonical-cleaned native surfaces are required")
        commits, position = row.get("original_position_commits"), row.get("position")
        if index == 0:
            _need(type(commits) is int and commits == 0 and position is None,
                  "Original sentinel must be canonical row0")
        else:
            _need(type(commits) is int and commits == 1 and _integer(position)
                  and type(row.get("target")) is str and type(row.get("replacement")) is str
                  and bool(row["replacement"]), "Each nonoriginal row must commit one original word position")
        identity = []
        for name in names:
            values = row.get(field_map[name])
            _need(type(values) in (list, tuple) and len(values) == native_context
                  and all(_integer(v) for v in values), "Exact complete native integer token field required")
            if name == "attention_mask":
                _need(all(v in (0, 1) for v in values), "Attention mask must remain binary")
            identity.append((name, tuple(values)))
        identity_tuple = tuple(identity)
        _need(identity_tuple not in identities, "Canonical native aliases must already be deduplicated")
        identities.add(identity_tuple)
        snapshots.append(PoolRow(index, raw_index, row["raw"], row["native"], position,
                                 row.get("target"), row.get("replacement"), commits,
                                 identity_tuple, _metadata_json(row)))
    original_words = snapshots[0].raw.split(" ")
    _need(all(bool(w) for w in original_words), "Original supplier words must remain nonempty")
    _need(all(r.position is None or r.position < len(original_words) for r in snapshots),
          "Candidate positions must refer to the original caption")
    covered, alias_json, overflow_json = set(reps), [], []
    for alias in aliases:
        _need(type(alias) is dict, "Explicit alias metadata required")
        raw_index, representative_index = alias.get("raw_index"), alias.get("representative_index")
        _need(_integer(raw_index) and raw_index < raw_count and raw_index not in covered
              and _integer(representative_index) and representative_index < len(reps)
              and reps[representative_index] < raw_index, "Stable first native alias mapping required")
        covered.add(raw_index)
        alias_json.append(_metadata_json(alias))
    for excluded in unencodable:
        _need(type(excluded) is dict and excluded.get("kind") == "native_context_overflow",
              "Only an already recorded native-context overflow may be represented")
        raw_index = excluded.get("raw_index")
        _need(_integer(raw_index, minimum=1) and raw_index < raw_count and raw_index not in covered
              and type(excluded.get("error")) is str
              and "is too long for context length" in excluded["error"],
              "Original overflow is fatal; preserve the exact historical overflow error")
        covered.add(raw_index)
        overflow_json.append(_metadata_json(excluded))
    _need(covered == set(range(raw_count)), "Representatives, aliases and permitted overflow must cover the raw supply")
    domain = dict(rows=[json.loads(r.original_metadata_json) for r in snapshots],
                  native_field_names=list(names), native_context=native_context,
                  representatives=list(reps), aliases=[json.loads(x) for x in alias_json],
                  unencodable=[json.loads(x) for x in overflow_json], raw_count=raw_count)
    return CanonicalPool(tuple(snapshots), names, native_context, reps, tuple(alias_json),
                         tuple(overflow_json), raw_count, _metadata_sha(domain))


def _checked_pool_group(owner: OwnerContract, pools: Any) -> tuple[CanonicalPool, ...]:
    owner = _checked_owner(owner)
    _need(type(pools) in (tuple, list) and len(pools) == len(owner.caption_ids),
          "Every caption in the complete m_i owner group needs its own pool")
    result = tuple(pools)
    for pool in result:
        _need(type(pool) is CanonicalPool and pool.native_context == owner.native_context
              and set(pool.native_field_names) == set(owner.native_fields),
              "Do not use another source's native fields or truncate policy")
        _need(not pool.unencodable_json or owner.source.startswith("clip_"),
              "Fusion context30 has no overflow omission in the registered interface")
    return result


def random_scale_seed(owner: OwnerContract, step: int, role: str) -> int:
    """Same owner integer, zero-based step and original role namespace as old R3."""
    owner = _checked_owner(owner)
    _need(type(step) is int and 0 <= step < STEPS
          and role in ("side", "scale:down", "scale:up", "noise:down", "noise:up"),
          "An original finite step and role are required")
    raw = f"{SEED_NAMESPACE}:{42 + owner.owner}:{step}:{role}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


@dataclass(frozen=True, slots=True, eq=False)
class ScaleDraw:
    slot: int
    scale: float
    native_size: tuple[int, int]
    scale_seed: int | None
    noise_seed: int | None
    noise: Any
    rho: float
    mode: str


class ScaleStreams:
    """Original local u/standard-normal role pairing, generalized carrier only."""

    def __init__(self, owner: OwnerContract, recipe: Recipe):
        self.owner, self.recipe = _checked_owner(owner), _checked_recipe(recipe)
        self.steps = 0

    def draw(self, shape: Any) -> tuple[ScaleDraw, ...]:
        _require_server_runtime()
        import numpy as np

        height = self.owner.carrier_size
        _need(tuple(shape) == (1, 3, height, height) and self.steps < STEPS,
              "One original carrier shape and at most ten draws are required")
        step, width = self.steps, self.recipe.rho
        mode = "no_scale" if width == 0.0 else "interval"
        slots = (1,) if self.recipe.path_mode == "point" else (0, 1, 2)
        draws = []
        for slot in slots:
            if slot == 1:
                draws.append(ScaleDraw(1, 1.0, (height, height), None, None, None, width, mode))
                continue
            side = "down" if slot == 0 else "up"
            if mode == "no_scale":
                scale, scale_seed = 1.0, None
            else:
                low, high = (1.0 - width, 1.0) if side == "down" else (1.0, 1.0 + width)
                scale_seed = random_scale_seed(self.owner, step, f"scale:{side}")
                sampled = float(np.random.RandomState(scale_seed).uniform(low, high))
                _need(np.isfinite(sampled) and low <= sampled <= high,
                      "Invalid original local scale sample; no resampling")
                scale = min(sampled, float(np.nextafter(high, low)))
            noise_seed = random_scale_seed(self.owner, step, f"noise:{side}")
            # NoNoise still draws the same complete standard-normal array and
            # multiplies by zero. NoScale skips scale draws but keeps both noise roles.
            noise = (np.random.RandomState(noise_seed).standard_normal(shape)
                     * self.recipe.sigma).astype(np.float32)
            # NEW SCIENTIFIC DEFINITION for rho=1; old positive sizes are unchanged.
            size = max(1, math.floor(scale * height))
            draws.append(ScaleDraw(slot, scale, (size, size), scale_seed, noise_seed,
                                   noise, width, mode))
        self.steps += 1
        return tuple(draws)


def _clean_tensor(clean: Any, owner: OwnerContract) -> None:
    _require_server_runtime()
    import torch

    height = owner.carrier_size
    _need(isinstance(clean, torch.Tensor) and clean.dtype == torch.float32
          and tuple(clean.shape) == (1, 3, height, height) and clean.is_leaf
          and not clean.requires_grad and bool(torch.isfinite(clean).all())
          and not bool((clean < 0).any()) and not bool((clean > 1).any()),
          "A detached finite native carrier FP32 pixel leaf in [0,1] is required")


def _checked_endpoint_unit(unit: Any, owner: OwnerContract, device: Any, *, differentiable: bool) -> None:
    _require_server_runtime()
    import torch

    _need(isinstance(unit, torch.Tensor) and unit.dtype == torch.float32
          and tuple(unit.shape) == (1, owner.dimension) and unit.device == device
          and unit.requires_grad is differentiable and bool(torch.isfinite(unit).all())
          and bool(((unit.detach().norm(dim=-1) - 1).abs() < UNIT_TOLERANCE).all()),
          "Unmodified finite same-source native FP32 near-unit endpoint required")


def collect_initial_reference(clean: Any, clean_unit: Any, endpoint: Any,
                              owner_contract: OwnerContract, recipe: Recipe,
                              *, on_view: Any = None) -> tuple[Any, dict[str, Any]]:
    """Reuse actual clean; optionally encode one direct carrier-grid BOX view.

    The caller's endpoint alone maps carrier pixels to the native model input.
    For fusion384 that endpoint must normalize the direct384 carrier, not use
    the historical local224-to384 resize. No BOX view is an attack-path view.
    """
    _require_server_runtime()
    import numpy as np
    import torch
    from torch.nn import functional as F

    owner, recipe = _checked_owner(owner_contract), _checked_recipe(recipe)
    _clean_tensor(clean, owner)
    _checked_endpoint_unit(clean_unit, owner, clean.device, differentiable=False)
    _need(callable(endpoint) and (on_view is None or callable(on_view)), "Own-source endpoint and optional observer required")
    _need(recipe.control in ("full", "no_box", "no_scale", "no_noise"),
          "Path/search controls must reuse the actual Full reference packet")

    def synchronize() -> None:
        if clean.is_cuda:
            torch.cuda.synchronize(clean.device)

    units, view_pins, labels = [], [], []
    convolution_seconds = encoder_seconds = transfer_seconds = 0.0
    with torch.no_grad():
        views = [("clean", clean)]
        if recipe.kernel != 1:
            synchronize()
            started = time.perf_counter()
            low, high = (recipe.kernel - 1) // 2, recipe.kernel // 2
            boxed = F.avg_pool2d(F.pad(clean, (low, high, low, high), mode="replicate"),
                                 kernel_size=recipe.kernel, stride=1, padding=0)
            synchronize()
            convolution_seconds = time.perf_counter() - started
            views.append((f"box{recipe.kernel}", boxed))
        for index, (label, pixels) in enumerate(views):
            if index == 0:
                unit = clean_unit
            else:
                synchronize()
                started = time.perf_counter()
                unit = endpoint(pixels)
                synchronize()
                encoder_seconds += time.perf_counter() - started
            _checked_endpoint_unit(unit, owner, clean.device, differentiable=False)
            synchronize()
            started = time.perf_counter()
            units.append(unit.detach().cpu().numpy()[0].copy())
            observed_pixels = pixels.detach().cpu().numpy().copy()
            view_pins.append(hashlib.sha256(observed_pixels.tobytes()).hexdigest())
            labels.append(label)
            if on_view is not None:
                on_view(label, observed_pixels)
            synchronize()
            transfer_seconds += time.perf_counter() - started
    return np.stack(units), dict(
        phase="initial_candidate_reference", owner=owner_metadata(owner), recipe=recipe_metadata(recipe),
        view_labels=labels, view_sha256=view_pins, native_sizes=[[owner.carrier_size] * 2 for _ in labels],
        clean_reuses_observed_clean_unit=True, direct_from_same_original_clean=True,
        BOX_applied=recipe.kernel != 1, kernel=recipe.kernel, cascade=False,
        stride=None if recipe.kernel == 1 else 1, padding=None if recipe.kernel == 1 else "replicate",
        padding_placement=None if recipe.kernel == 1 else "SAME_UPPER", same_size=True,
        extra_selection_forwards=0 if recipe.kernel == 1 else 1, selection_VJP=0,
        initial_extra_normalization=False, initial_RGB8_quantization=False,
        initial_random_sampling=False, advances_image_path_random_stream=False,
        convolution_seconds=convolution_seconds, initial_encoder_seconds=encoder_seconds,
        selection_evidence_transfer_seconds=transfer_seconds,
        observed_endpoint_origin_requires_external_server_evidence=True,
    )


def _unit_matrix(value: Any, rows: int | None, dimension: int) -> Any:
    _require_server_runtime()
    import numpy as np

    matrix = np.asarray(value)
    _need(matrix.dtype == np.float32 and matrix.ndim == 2 and matrix.shape[1] == dimension
          and matrix.shape[0] > 0 and (rows is None or matrix.shape[0] == rows)
          and bool(np.isfinite(matrix).all())
          and float(np.max(np.abs(np.linalg.norm(matrix.astype(np.float64), axis=1) - 1))) <= UNIT_TOLERANCE,
          "Complete unchanged native FP32 near-unit rows required; no repair/renormalization")
    return matrix


def _frozen_copy(value: Any) -> Any:
    _require_server_runtime()
    result = value.copy()
    result.setflags(write=False)
    return result


def _array_pin(value: Any) -> str:
    _require_server_runtime()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_metadata_json(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class PreparedFull:
    owner: OwnerContract
    recipe: Recipe
    pools: tuple[CanonicalPool, ...]
    text_units: tuple[Any, ...]
    image_units: Any
    reference64: Any
    reference_sha256: str
    scores: tuple[Any, ...]
    original_units: Any
    domain_sha256: str
    preparation_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class OwnerSelection:
    prepared: PreparedFull
    recipe: Recipe
    indices: tuple[int, ...]
    selected_units: Any
    guide_cpu: Any
    trace_json: str


def _build_selection(prepared: PreparedFull, recipe: Recipe, indices: Any,
                     trace: dict[str, Any]) -> OwnerSelection:
    _require_server_runtime()
    import numpy as np

    _need(type(indices) in (tuple, list) and len(indices) == len(prepared.owner.caption_ids),
          "One selected canonical row for every owner caption required")
    chosen = tuple(indices)
    _need(all(_integer(i) and i < len(p.rows) for i, p in zip(chosen, prepared.pools, strict=True)),
          "One same-pool canonical row index required")
    selected = np.stack([p[i].copy() for p, i in zip(prepared.text_units, chosen, strict=True)])
    guide_rows = prepared.original_units if recipe.guide_mode == "original_o" else selected
    # All m_i rows, FP32 -> CPU FP64 mean, no post-mean normalization.
    guide = guide_rows.astype(np.float64).mean(axis=0)
    _need(guide.shape == (prepared.owner.dimension,) and bool(np.isfinite(guide).all()),
          "Finite complete-owner CPU FP64 guide required")
    trace = dict(trace, owner=owner_metadata(prepared.owner), recipe=recipe_metadata(recipe),
                 preparation_sha256=prepared.preparation_sha256,
                 domain_sha256=prepared.domain_sha256, actual_reference_sha256=prepared.reference_sha256,
                 indices=list(chosen), fixed_Q_count=len(chosen), guide_count=len(guide_rows),
                 guide_policy="complete_owner_native_FP32_rows_to_CPU_FP64_mean_unnormalized",
                 guide_renormalized=False, selected_units_renormalized=False,
                 fixed_Q_delivery=True, selection_repeated_each_step=False,
                 arithmetic_is_not_independent_origin_acceptance=True)
    return OwnerSelection(prepared, recipe, chosen, _frozen_copy(selected),
                          _frozen_copy(guide), _metadata_json(trace))


def prepare_full_selection(owner_contract: OwnerContract, recipe: Recipe,
                           canonical_pools: Any, text_units: Any, image_units: Any) -> OwnerSelection:
    """Complete FP64 first-argmin per caption, preserving original/aliases/order.

    Read-only private snapshots of the actual owner feature matrices are made
    for later same-pool controls. This extra CPU copying/retention is explicit
    overhead to account in a future worker; it is not described as free.
    """
    _require_server_runtime()
    import numpy as np

    owner, recipe = _checked_owner(owner_contract), _checked_recipe(recipe)
    _need(recipe.control in ("full", "no_box", "no_scale", "no_noise"),
          "Importance/O-guide/POINT/NoMomentum derive from the actual Full packet")
    pools = _checked_pool_group(owner, canonical_pools)
    _need(type(text_units) in (tuple, list) and len(text_units) == len(pools),
          "Each complete own-caption pool needs its native feature matrix")
    features = tuple(_frozen_copy(_unit_matrix(v, len(p.rows), owner.dimension))
                     for p, v in zip(pools, text_units, strict=True))
    images = _frozen_copy(_unit_matrix(image_units, 1 if recipe.kernel == 1 else 2, owner.dimension))
    reference = _frozen_copy(images.astype(np.float64).mean(axis=0))
    scores, indices = [], []
    for matrix in features:
        values = matrix.astype(np.float64) @ reference
        _need(bool(np.isfinite(values).all()), "Nonfinite candidate score; no fallback")
        indices.append(int(np.argmin(values)))
        scores.append(_frozen_copy(values))
    originals = _frozen_copy(np.stack([matrix[0].copy() for matrix in features]))
    domain_sha = _metadata_sha([p.domain_sha256 for p in pools])
    reference_sha = _array_pin(reference)
    preparation_sha = _metadata_sha(dict(
        owner=owner_metadata(owner), recipe=recipe_metadata(recipe), domain_sha256=domain_sha,
        text_unit_pins=[_array_pin(m) for m in features], image_unit_pin=_array_pin(images),
        actual_reference_sha256=reference_sha, score_pins=[_array_pin(s) for s in scores]))
    prepared = PreparedFull(owner, recipe, pools, features, images, reference, reference_sha,
                            tuple(scores), originals, domain_sha, preparation_sha)
    return _build_selection(prepared, recipe, indices, dict(
        selector="complete_canonical_pool_CPU_FP64_first_argmin",
        candidate_counts=[len(p.rows) for p in pools],
        original_sentinel_eligible=True, canonical_aliases_restored=False,
        candidate_scores_repaired=False, same_dtype_native_units=True,
        private_CPU_feature_snapshots_nonzero_memory_and_copy_cost=True))


def delivery_metadata(selection: OwnerSelection) -> dict[str, Any]:
    """Return the selected literal canonical surfaces, not an abstracted guide."""
    _need(type(selection) is OwnerSelection, "An explicit owner selection is required")
    rows = []
    for caption_id, pool, index in zip(selection.prepared.owner.caption_ids,
                                       selection.prepared.pools, selection.indices, strict=True):
        row = pool.rows[index]
        rows.append(dict(caption_id=caption_id, index=index, raw_index=row.raw_index,
                         raw=row.raw, native=row.native, position=row.position,
                         original_position_commits=row.original_position_commits,
                         canonical_metadata=json.loads(row.original_metadata_json)))
    return dict(schema="CSGA_LITERAL_Q_DELIVERY_METADATA_V1", core_version=CORE_VERSION,
                owner=owner_metadata(selection.prepared.owner), recipe=recipe_metadata(selection.recipe),
                rows=rows, selection_trace=json.loads(selection.trace_json),
                selected_literal_Q_used_for_delivery=True, no_semantic_abstraction=True,
                clean_caption_and_PNG_readback_are_external_caller_duties=True)
