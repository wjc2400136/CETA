"""Explicit SADCA 2x5 adapter with owner-serial original-API RNG scheduling.

No model/data loader, CLI, process launcher, seed or file writer lives here.
The caller supplies the unchanged pinned SADCA and its complete clean pool.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
import math
import pickle
import random
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .source_bridge import verify_native_source_bridge


BASELINE_SHA256 = "1266492a2da96c5432a69f573fdd5b556895a8a95268bc5f56efa052d295cd51"
WITNESS_SHA256 = "07a68d17bb4f0dc0708951e2705730997e34a0456afe2e703f64105a488dd99d"
VARIANT = "sadca_explicit_2x5_original_api_tape_nativeB4_server_sources_suite_task_v27_native4_owner_model_microbatch_v4"
_LOCK = threading.RLock()
_OPS = ("vertical_shift", "horizontal_shift", "vertical_flip", "horizontal_flip",
        "rotate180", "scale", "add_noise", "BSR")
_TRANSFORMS = (*_OPS, "get_length", "shuffle_single_dim", "local_semantic_augmentation",
               "mix_texts_in_groups")


class FidelityError(RuntimeError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass
class RNGState:
    python: object
    numpy: tuple
    cpu: torch.Tensor
    cuda_initialized: bool
    cuda: tuple

    def hashes(self):
        return dict(python=_sha(pickle.dumps(self.python, protocol=4)),
                    numpy=_sha(pickle.dumps(self.numpy, protocol=4)),
                    torch_cpu=_sha(self.cpu.numpy().tobytes()),
                    cuda_initialized=self.cuda_initialized,
                    torch_cuda=[_sha(x.cpu().numpy().tobytes()) for x in self.cuda])


def capture_rng_state():
    initialized = torch.cuda.is_initialized()
    return RNGState(copy.deepcopy(random.getstate()), copy.deepcopy(np.random.get_state()),
                    torch.get_rng_state().clone(), initialized,
                    tuple(x.clone() for x in torch.cuda.get_rng_state_all()) if initialized else ())


def restore_rng_state(state):
    if torch.cuda.is_initialized() != state.cuda_initialized:
        raise FidelityError("CUDA initialization changed across the RNG boundary")
    random.setstate(copy.deepcopy(state.python))
    np.random.set_state(copy.deepcopy(state.numpy))
    torch.set_rng_state(state.cpu.clone())
    if state.cuda_initialized:
        torch.cuda.set_rng_state_all([x.clone() for x in state.cuda])


def _assert_rng(expected, where):
    if capture_rng_state().hashes() != expected.hashes():
        raise FidelityError("Unexpected RNG consumption in " + where)


def _code_shape(code):
    return dict(code=code.co_code.hex(),
                consts=[_code_shape(x) if isinstance(x, types.CodeType)
                        else marshal.dumps(x).hex() for x in code.co_consts],
                names=code.co_names, vars=code.co_varnames, free=code.co_freevars,
                cell=code.co_cellvars, args=code.co_argcount,
                posonly=code.co_posonlyargcount, kwonly=code.co_kwonlyargcount,
                flags=code.co_flags, exception_table=getattr(code, "co_exceptiontable", b"").hex())


def _code_hash(code):
    return _sha(json.dumps(_code_shape(code), sort_keys=True).encode())


def _class_code(module, name):
    return next(x for x in module.co_consts if isinstance(x, types.CodeType) and x.co_name == name)


def _method_check(obj, name, cls):
    method = getattr(obj, name)
    expected = next(x for x in cls.co_consts if isinstance(x, types.CodeType) and x.co_name == name)
    if not isinstance(method, types.MethodType) or _code_hash(method.__func__.__code__) != _code_hash(expected):
        raise FidelityError("Pinned SADCA method was replaced: " + name)


def _verify(attacker, baseline_path=None, *, allowed_inner=(5,), allowed_rounds=(2,)):
    path = Path(baseline_path) if baseline_path else Path(inspect.getsourcefile(type(attacker)) or "")
    if not path.is_file() or _sha(path.read_bytes()) != BASELINE_SHA256:
        raise FidelityError("SADCA baseline source bytes differ")
    path = path.resolve()
    witness_path = path.parent.parent / "sadca_word_witness.py"
    if not witness_path.is_file() or _sha(witness_path.read_bytes()) != WITNESS_SHA256:
        raise FidelityError("Pinned native source-word witness bytes differ")
    code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    ia, ta, native = attacker.image_attacker, attacker.text_attacker, attacker.native_attacker
    for name in ("attack", "_validated_pool", "_sample_negative_embeddings"):
        _method_check(attacker, name, _class_code(code, "SADCA"))
    for name in (*_TRANSFORMS, "loss_func_old", "loss_func_contrast", "img_pos", "txt_guided_attack"):
        _method_check(ia, name, _class_code(code, "ImageAttacker"))
    for name in ("img_guided_attack", "loss_func", "_tokenize", "_get_masked", "get_important_scores"):
        _method_check(ta, name, _class_code(code, "TextAttacker"))
    _method_check(native, "attack", _class_code(code, "Attacker"))
    globals_ = ia.local_semantic_augmentation.__func__.__globals__
    for name in ("get_substitues", "get_bpe_substitues"):
        expected = next(x for x in code.co_consts if isinstance(x, types.CodeType) and x.co_name == name)
        fn = ta.img_guided_attack.__func__.__globals__.get(name)
        if not isinstance(fn, types.FunctionType) or _code_hash(fn.__code__) != _code_hash(expected):
            raise FidelityError("Pinned text candidate helper differs: " + name)
    bridge_receipt = verify_native_source_bridge(attacker.model_name, attacker.model, attacker.tokenizer)
    suite_budget = (
        isinstance(attacker.num_perturb, int) and not isinstance(attacker.num_perturb, bool)
        and attacker.num_perturb in range(1, 6)
        and any(math.isclose(attacker.epsilon, pixel / 255, rel_tol=0.0, abs_tol=1e-15)
                for pixel in (1, 2, 4, 8, 16))
        and math.isclose(attacker.step_size, attacker.epsilon / 4, rel_tol=0.0, abs_tol=1e-15)
    )
    fixed = (str(attacker.model_name).lower() in {"albef", "tcl", "clip_vit_b16", "clip_rn101"}
             and suite_budget and attacker.steps == 10
             and type(ia.steps) is int and ia.steps in allowed_inner
             and attacker.epsilon == ia.eps
             and attacker.step_size == ia.step_size
             and attacker.max_length == ia.max_length == ta.max_length == 30
             and attacker.topk_neg == 20 and attacker.num_perturb == ta.num_perturbation
             and ia.input_trans_num == 10 and ia.num_block == 3 and ia.sample_numbers == 5
             and ia.lam == ta.lam == native.lam == 0.2
             and type(native.dynamic_num) is int and native.dynamic_num in allowed_rounds
             and ta.cls is False and ta.topk == 10 and ta.batch_size == 32
             and ta.threshold_pred_score == 0.3
             and native.model is attacker.model and native.img_attacker is ia
             and native.txt_attacker is ta and native.tokenizer is attacker.tokenizer
             and ta.tokenizer is attacker.tokenizer and ta.ref_net is attacker.ref_model
             and ia.normalization is attacker.normalization and ia.device == attacker.device)
    if not fixed:
        raise FidelityError("Only pinned native source/suite SADCA with the explicit two-round five-step variant is admitted")
    if len(ia.op) != len(_OPS) or any(
            not isinstance(fn, types.MethodType) or fn.__self__ is not ia
            or fn.__func__ is not getattr(ia, name).__func__ for fn, name in zip(ia.op, _OPS, strict=True)):
        raise FidelityError("The native ordered augmentation operator list differs")
    attacker._sadca_native_source_bridge_receipt = bridge_receipt
    return path, globals_


def configure_variant(attacker, steps, *, baseline_path=None):
    """After canonical construction, apply the two declared 2x5 overrides once."""
    if type(steps) is not int or steps != 5:
        raise ValueError("Only the explicitly authorized two-round five-step variant exists")
    if hasattr(attacker, "_sadca_batch_variant_receipt"):
        raise FidelityError("Configure a fresh canonical attacker once; no repeated override")
    path, _ = _verify(attacker, baseline_path, allowed_inner=(10,), allowed_rounds=(5,))
    receipt = dict(kind="SADCA_EXPLICIT_2X5_VARIANT_V1", baseline_path=str(path),
                   baseline_sha256=BASELINE_SHA256, canonical_constructor_steps=10,
                   wrapper_steps_retained=attacker.steps, actual_inner_steps=steps,
                   dynamic_rounds=2, final_image_updates=2 * steps, alignment_auxiliary_updates=5,
                   epsilon=attacker.epsilon, step_size=attacker.step_size,
                   numerical_overrides={"image_attacker.steps": dict(previous=10, value=steps),
                       "native_attacker.dynamic_num": dict(previous=5, value=2)},
                   original_source_files_modified=False, factory_guards_modified=False)
    attacker.image_attacker.steps = steps
    attacker.native_attacker.dynamic_num = 2
    attacker._sadca_batch_variant_receipt = copy.deepcopy(receipt)
    return receipt


class _ShapeTensor:
    """Dimensions only; deterministic tensor transformations do no arithmetic."""
    def __init__(self, shape, device):
        self.shape, self.device = tuple(int(x) for x in shape), torch.device(device)

    def clone(self):
        return _ShapeTensor(self.shape, self.device)

    def size(self, dim):
        return self.shape[dim]

    def roll(self, *args, **kwargs):
        return self.clone()

    def flip(self, *args, **kwargs):
        return self.clone()

    def rot90(self, *args, **kwargs):
        return self.clone()  # The pinned operator is exactly k=2.

    def __add__(self, other):
        return self.clone()

    __radd__ = __add__

    def __mul__(self, other):
        return self.clone()

    __rmul__ = __mul__

    def split(self, lengths, dim):
        if sum(lengths) != self.shape[dim] or any(x < 0 for x in lengths):
            raise FidelityError("Native BSR shape partition differs")
        parts = []
        for length in lengths:
            shape = list(self.shape)
            shape[dim] = int(length)
            parts.append(_ShapeTensor(shape, self.device))
        return tuple(parts)

    def uniform_(self, *args, **kwargs):
        # Preserve the original DEVICE and shape of zeros_like(x).uniform_.
        # This draws real CPU/CUDA RNG values, but performs no image transform/model call.
        torch.zeros(self.shape, device=self.device, dtype=torch.float32).uniform_(*args, **kwargs)
        return self


class _ShapeScalar:
    def __getitem__(self, index):
        return self

    def __mul__(self, value):
        return value.clone()


class _ShapeTorch:
    @staticmethod
    def rand(*args, **kwargs):
        torch.rand(*args, **kwargs)  # Exact native CPU torch.rand(1) API.
        return _ShapeScalar()

    @staticmethod
    def zeros_like(value):
        return value.clone()

    @staticmethod
    def clip(value, *args, **kwargs):
        return value

    @staticmethod
    def cat(values, dim=0):
        shape = list(values[0].shape)
        shape[dim] = sum(x.shape[dim] for x in values)
        if any(any(x.shape[i] != shape[i] for i in range(len(shape)) if i != dim) for x in values):
            raise FidelityError("Native BSR concatenation shape differs")
        return _ShapeTensor(shape, values[0].device)


class _ShapeTransforms:
    def __init__(self, original):
        self.original = original

    def RandomResizedCrop(self, *args, **kwargs):
        transform = self.original.RandomResizedCrop(*args, **kwargs)

        def crop(value):
            # get_params inspects dimensions only. Keep its actual rejection loop,
            # CPU random API, ratio, fallback and call order; do not replace with a seed.
            dimensions_only = torch.empty(value.shape, dtype=torch.float32, device="cpu")
            transform.get_params(dimensions_only, transform.scale, transform.ratio)
            shape = (*value.shape[:-2], *transform.size)
            return _ShapeTensor(shape, value.device)
        return crop


def _shape_helper(ia, globals_):
    helper = copy.copy(ia)
    rewritten = dict(globals_)
    rewritten.update(torch=_ShapeTorch(), transforms=_ShapeTransforms(globals_["transforms"]))
    for name in _TRANSFORMS:
        original = getattr(ia, name).__func__
        fn = types.FunctionType(original.__code__, rewritten, original.__name__,
                                original.__defaults__, original.__closure__)
        fn.__kwdefaults__ = copy.copy(original.__kwdefaults__)
        setattr(helper, name, types.MethodType(fn, helper))
    helper.op = [getattr(helper, name) for name in _OPS]
    return helper


@dataclass
class _Phase:
    label: str
    before: RNGState
    after: RNGState

    def metadata(self):
        return dict(label=self.label, before=self.before.hashes(), after=self.after.hashes())


def _record_phase(label, fn):
    before = capture_rng_state()
    fn()
    return _Phase(label, before, capture_rng_state())


def _replay(phase, live, fn):
    _assert_rng(live, "before replay " + phase.label)
    try:
        restore_rng_state(phase.before)
        result = fn()
        _assert_rng(phase.after, "original API replay " + phase.label)
        return result
    finally:
        restore_rng_state(live)


def _record_owner(ia, globals_, shape, texts, pool, owner_id, dynamic_rounds, negative_self_mask_size=None):
    """Consume one complete ORIGINAL owner's RNG without any model/text search."""
    helper = _shape_helper(ia, globals_)
    phases = {}
    m = len(texts)
    # Retrieval: all current positives are this physical image's self-mask.
    # Task: the original task bridge masks all physical-image texts, while
    # retaining only this annotation's positives in text search and losses.
    mask_size = m if negative_self_mask_size is None else negative_self_mask_size
    if type(mask_size) is not int or not 1 <= mask_size < len(pool["text_features"]):
        raise ValueError("Invalid complete physical-image negative self-mask size")

    def negatives():
        for _ in texts:
            torch.randperm(len(pool["image_features"]) - 1)
        torch.randperm(len(pool["text_features"]) - mask_size)

    def noise():
        np.random.uniform(-ia.eps, ia.eps, shape)

    def transformed_and_mix():
        point = _ShapeTensor(shape, ia.device)
        for _ in range(ia.input_trans_num - 1):
            helper.local_semantic_augmentation(point)
        helper.mix_texts_in_groups(texts, [0] * m)
        # Actual adversarial strings are deliberately NOT guessed. In the pinned
        # concat branch shuffle consumes randomness solely as a function of m.
        helper.mix_texts_in_groups(["parameter-tape-slot"] * m, [0] * m)

    phases["negative"] = _record_phase("owner:%s/negative" % owner_id, negatives)
    phases["alignment_noise"] = _record_phase("owner:%s/alignment_noise" % owner_id, noise)
    for outer in range(dynamic_rounds):
        phases[(outer, "noise")] = _record_phase("owner:%s/round:%s/noise" % (owner_id, outer), noise)
        for step in range(ia.steps):
            phases[(outer, step)] = _record_phase(
                "owner:%s/round:%s/step:%s/views_and_two_text_permutations" % (owner_id, outer, step),
                transformed_and_mix)
    return phases


def _noise(ia, shape, device):
    return torch.from_numpy(np.random.uniform(-ia.eps, ia.eps, shape)).float().to(device)


def _update(ia, current, gradient, momentum, clean):
    normalized = gradient / torch.mean(torch.abs(gradient), dim=(1, 2, 3), keepdim=True)
    momentum = momentum * 1.0 + normalized
    after = current.detach() + ia.step_size * momentum.sign()
    after = torch.min(torch.max(after, clean - ia.eps), clean + ia.eps)
    return torch.clamp(after, 0.0, 1.0), momentum


def attack_batch4(attacker, images, text_groups, image_ids, text_id_groups, pool,
                  trace_hook=None, *, baseline_path=None, owner_ids=None,
                  task_source_index=None, task_row_ids=None, sample_ids=None):
    """Return (NCHW pixels, complete caption groups, observations), N=1..4.

    One exclusive process owns the Python/NumPy/Torch RNG streams. All tape
    generation, real random draws/replay and checks belong in measured GEN.
    Models/tokenizers must consume no RNG; violations and OOM propagate with
    an attached partial audit, without a lower-batch fallback or retry.
    """
    with _LOCK:
        started = time.perf_counter()
        _source_path, globals_ = _verify(attacker, baseline_path)
        config = getattr(attacker, "_sadca_batch_variant_receipt", None)
        if (not config or config["actual_inner_steps"] != attacker.image_attacker.steps
                or config["dynamic_rounds"] != attacker.native_attacker.dynamic_num):
            raise FidelityError("Call configure_variant once before the native/B4 run")
        groups = list(text_groups)
        if any(isinstance(x, (str, bytes)) for x in groups):
            raise ValueError("Expected owner-grouped complete caption sequences")
        groups = [list(x) for x in groups]
        integer_id = globals_["_integer_id"]
        ids = [integer_id(x) for x in image_ids]
        caption_ids = [[integer_id(x) for x in group] for group in text_id_groups]
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.dtype != torch.float32
                or not 1 <= len(images) <= 4 or images.shape[1] != 3
                or len(groups) != len(images) or len(ids) != len(images) or len(caption_ids) != len(images)
                or any(not g or any(not isinstance(x, str) for x in g) for g in groups)):
            raise ValueError("Require N=1..4 FP32 RGB images with complete positive groups")
        task_mode = task_source_index is not None
        bindings = []
        if task_mode:
            raise ValueError("This release implements retrieval only")
        else:
            if task_row_ids is not None or sample_ids is not None:
                raise ValueError("Incomplete task binding cannot use the retrieval path")
            if len(set(ids)) != len(ids):
                raise ValueError("Retrieval physical image IDs must be unique within the batch")
            view = attacker._validated_pool(pool)
            for physical, text_ids, texts in zip(ids, caption_ids, groups, strict=True):
                if (physical not in view["image_rows"]
                        or tuple(text_ids) != view["image_to_captions"][physical]
                        or len(text_ids) != len(texts)):
                    raise ValueError("Retrieval owner must retain all frozen pool caption IDs in order")
            negative_self_masks = copy.deepcopy(caption_ids)
            logical_ids = ids if owner_ids is None else list(owner_ids)
            if (len(logical_ids) != len(groups) or len(set(logical_ids)) != len(logical_ids)
                    or any(type(value) not in (int, str) for value in logical_ids)):
                raise ValueError("External owner identity is metadata-only and must be complete and unique")
        images = images.detach().to(attacker.device)
        ia, ta, model = attacker.image_attacker, attacker.text_attacker, attacker.model
        model.eval()
        attacker.ref_model.eval()
        flags = tuple(p.requires_grad for p in model.parameters())
        initial = capture_rng_state()
        audit = dict(kind="SADCA_TRUE_OWNER_BATCH4_ACTUAL_AUDIT_V1", state="STARTED",
                     variant=copy.deepcopy(config), adapter_variant=VARIANT, owner_ids=logical_ids,
                     physical_image_ids=ids, task_bindings=copy.deepcopy(bindings),
                     negative_self_mask_caption_ids=copy.deepcopy(negative_self_masks),
                     source_bridge_receipt=copy.deepcopy(attacker._sadca_native_source_bridge_receipt),
                     caption_ids=caption_ids, owner_count=len(groups), alignment_forward_batches=[],
                     dynamic_forward_batches=[], backward_calls=0, gradient_owner_blocks=0,
                     original_negative_preparation_calls=0, tape_model_calls=0,
                     negative_selections=[], final_source_word_witnesses_by_owner=[], round_source_word_witnesses=[],
                     rng_initial_hashes=initial.hashes(), model_trainability_changed=False,
                     RNG_tape_includes_device_uniform_draw_cost=True, scientific_acceptance=False, model_forward_microbatch="owner_original_1_alignment_10_dynamic", optimizer_tensor_batch=len(groups), one_backward_per_update=True, B1_attack_fallback=False)
        n = len(groups)
        try:
            tapes = [_record_owner(ia, globals_, (1, *images.shape[1:]), texts, view, owner,
                                   attacker.native_attacker.dynamic_num, len(self_mask))
                     for texts, owner, self_mask in zip(groups, logical_ids, negative_self_masks, strict=True)]
            live = capture_rng_state()
            audit["rng_after_tape_hashes"] = live.hashes()
            summaries = []
            for owner, tape in zip(logical_ids, tapes, strict=True):
                phase_metadata = [phase.metadata() for phase in tape.values()]
                summaries.append(dict(owner_id=owner, phase_count=len(phase_metadata),
                    original_API_phase_order_sha256=_sha(json.dumps(phase_metadata, sort_keys=True).encode()),
                    initial_rng_hashes=phase_metadata[0]["before"], final_rng_hashes=phase_metadata[-1]["after"]))
            audit["rng_owner_phase_summaries"] = summaries
            neg_images, neg_texts, original_features = [], [], []

            def checked_call(label, fn):
                value = fn()
                _assert_rng(live, label)
                return value

            for i, texts in enumerate(groups):
                neg_i, neg_t = _replay(tapes[i]["negative"], live, lambda i=i, texts=texts:
                    attacker._sample_negative_embeddings(images[i:i+1], texts, [0] * len(texts),
                                                         [ids[i]], [negative_self_masks[i]], view))
                audit["original_negative_preparation_calls"] += 1
                audit["negative_selections"].append(copy.deepcopy(attacker.last_negative_selection))
                neg_images.append(neg_i)
                neg_texts.append(neg_t)
                with torch.no_grad():
                    tokens = checked_call("original tokenizer", lambda texts=texts: ta.tokenizer(
                        texts, padding="max_length", truncation=True, max_length=30, return_tensors="pt").to(attacker.device))
                    original_features.append(checked_call("original text encoding", lambda: model.inference_text(tokens))["text_feat"])
            alignment_noise = [_replay(tape["alignment_noise"], live,
                lambda: _noise(ia, (1, *images.shape[1:]), attacker.device)) for tape in tapes]
            col = torch.clamp(images.detach() + torch.cat(alignment_noise), 0.0, 1.0)
            alignment_momenta = [0] * n
            for step in range(5):
                col.requires_grad_()
                normalized = [ia.normalization(col[i:i+1]) if ia.normalization is not None else col[i:i+1] for i in range(n)]
                features = [model.inference_image(normalized[i])["image_feat"] for i in range(n)]
                model.zero_grad()
                loss = torch.tensor(0.0, dtype=torch.float32).to(attacker.device)
                for i, texts in enumerate(groups):
                    loss += -ia.loss_func_old(features[i], original_features[i], [0] * len(texts))
                loss.backward()
                gradient = col.grad
                next_rows = []
                for i in range(n):
                    row, alignment_momenta[i] = _update(ia, col[i:i+1], gradient[i:i+1], alignment_momenta[i], images[i:i+1])
                    next_rows.append(row)
                after = torch.cat(next_rows)
                audit["alignment_forward_batches"].extend([1] * n)
                audit["backward_calls"] += 1
                audit["gradient_owner_blocks"] += n
                _assert_rng(live, "alignment forward/backward/update")
                if trace_hook is not None:
                    trace_hook(dict(phase="alignment_step", step=step, owner_ids=logical_ids,
                                    before=col.detach(), raw_gradients=gradient.detach(), after=after.detach(),
                                    momentum=torch.cat(alignment_momenta).detach()))
                    _assert_rng(live, "alignment trace hook")
                col = after
            with torch.no_grad():
                aligned_features = [checked_call("aligned reference encoding", lambda i=i: model.inference_image(ia.normalization(col[i:i+1])))["image_feat"] for i in range(n)]
            aligned_supervision = [aligned_features[i][[0] * len(texts)] for i, texts in enumerate(groups)]
            adv = images.clone()  # Native Attacker.attack: alignment does NOT initialize final adv_imgs.
            momenta = [0] * n
            adversarial_groups = None
            for outer in range(attacker.native_attacker.dynamic_num):
                with torch.no_grad():
                    current_features = [checked_call("dynamic reference encoding", lambda i=i: model.inference_image(ia.normalization(adv[i:i+1])))["image_feat"] for i in range(n)]
                adversarial_groups, round_witnesses = [], []
                for i, texts in enumerate(groups):
                    adverse = checked_call("data-dependent original text search", lambda i=i, texts=texts:
                        ta.img_guided_attack(model, texts, img_embeds=aligned_supervision[i],
                            adv_img_embeds=current_features[i][[0] * len(texts)],
                            txt_embeds=original_features[i], neg_img_embeds=neg_images[i]))
                    adversarial_groups.append(list(adverse))
                    round_witnesses.append(copy.deepcopy(ta.last_source_word_witnesses))
                    # Retain the native otherwise-overwritten adv_txt_supervisions forward.
                    with torch.no_grad():
                        tokens = checked_call("adversarial tokenizer", lambda adverse=adverse: ta.tokenizer(
                            adverse, padding="max_length", truncation=True, max_length=30, return_tensors="pt").to(attacker.device))
                        checked_call("adversarial text encoding", lambda: model.inference_text(tokens))
                audit["round_source_word_witnesses"].append(round_witnesses)
                noise_rows = [_replay(tape[(outer, "noise")], live,
                    lambda: _noise(ia, (1, *images.shape[1:]), attacker.device)) for tape in tapes]
                adv = torch.clamp(adv.detach() + torch.cat(noise_rows), 0.0, 1.0)
                for step in range(ia.steps):
                    adv.requires_grad_()
                    owner_views, mixed_originals, mixed_adversarials = [], [], []
                    for i, texts in enumerate(groups):
                        def transform_and_mix(i=i, texts=texts):
                            point = adv if n == 1 else adv[i:i+1]
                            transformed = torch.cat([point] + [ia.local_semantic_augmentation(point)
                                                        for _ in range(ia.input_trans_num - 1)], 0)
                            return (transformed, ia.mix_texts_in_groups(texts, [0] * len(texts)),
                                    ia.mix_texts_in_groups(adversarial_groups[i], [0] * len(texts)))
                        transformed, mixed, mixed_adv = _replay(tapes[i][(outer, step)], live, transform_and_mix)
                        owner_views.append(transformed)
                        with torch.no_grad():
                            for values, destination in ((mixed, mixed_originals), (mixed_adv, mixed_adversarials)):
                                tokens = checked_call("mixed tokenizer", lambda values=values: ta.tokenizer(
                                    values, padding="max_length", truncation=True, max_length=30, return_tensors="pt").to(attacker.device))
                                destination.append(checked_call("mixed text encoding", lambda: model.inference_text(tokens))["text_feat"])
                    views = owner_views[0] if n == 1 else torch.cat([owner_views[i][v:v+1] for v in range(10) for i in range(n)], 0)
                    normalized = [ia.normalization(owner_views[i]) if ia.normalization is not None else owner_views[i] for i in range(n)]
                    features = [model.inference_image(normalized[i])["image_feat"] for i in range(n)]
                    model.zero_grad()
                    owner_losses = []
                    for i, texts in enumerate(groups):
                        loss_i = torch.tensor(0.0, dtype=torch.float32).to(attacker.device)
                        for v in range(10):
                            feature = features[i][v:v+1]  # Unused observation alias; loss retains TWO independent slices.
                            # Original negative term appears in BOTH calls; do not deduplicate.
                            loss_i += (ia.loss_func_contrast(features[i][v:v+1], mixed_adversarials[i], [0] * len(texts), neg_texts[i])
                                       + ia.loss_func_contrast(features[i][v:v+1], mixed_originals[i], [0] * len(texts), neg_texts[i]))
                        owner_losses.append(loss_i)
                    loss = owner_losses[0] if n == 1 else torch.tensor(0.0, dtype=torch.float32).to(attacker.device)
                    for loss_i in (() if n == 1 else owner_losses):
                        loss += loss_i
                    loss.backward()
                    gradient = adv.grad
                    rows = []
                    for i in range(n):
                        row, momenta[i] = _update(ia, adv[i:i+1], gradient[i:i+1], momenta[i], images[i:i+1])
                        rows.append(row)
                    after = torch.cat(rows)
                    audit["dynamic_forward_batches"].extend([10] * n)
                    audit["backward_calls"] += 1
                    audit["gradient_owner_blocks"] += n
                    _assert_rng(live, "dynamic forward/backward/update")
                    if trace_hook is not None:
                        trace_hook(dict(phase="dynamic_step", outer_round=outer, step=step, owner_ids=logical_ids,
                            before=adv.detach(), view_major_images=views.detach(), raw_gradients=gradient.detach(),
                            after=after.detach(), momentum=torch.cat(momenta).detach(),
                            owner_positive_and_negative_losses=tuple(x.detach() for x in owner_losses)))
                        _assert_rng(live, "dynamic trace hook")
                    adv = after
            audit["final_source_word_witnesses_by_owner"] = audit["round_source_word_witnesses"][-1]
            if flags != tuple(p.requires_grad for p in model.parameters()):
                raise FidelityError("Source model parameter trainability changed")
            _assert_rng(live, "completed SADCA batch")
            audit.update(state="RETURNED", rng_final_hashes=capture_rng_state().hashes(),
                         image_gradient_view_samples=n * (5 + attacker.native_attacker.dynamic_num * ia.steps * 10),
                         final_image_updates_per_owner=attacker.native_attacker.dynamic_num * ia.steps,
                         auxiliary_alignment_updates_per_owner=5,
                         text_search_calls=attacker.native_attacker.dynamic_num*n,
                         adapter_total_wall_seconds_without_outer_sync=time.perf_counter()-started)
            attacker.last_batch4_observation = audit
            return adv.detach(), adversarial_groups, audit
        except BaseException as error:
            audit.update(state="FAILED_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error),
                         adapter_total_wall_seconds_without_outer_sync=time.perf_counter()-started)
            try:
                error.sadca_batch4_audit = audit
            except Exception:
                pass
            raise
