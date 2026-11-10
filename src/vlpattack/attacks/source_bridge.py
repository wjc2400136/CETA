from __future__ import annotations


import hashlib


import inspect


import json


import marshal


import types


from pathlib import Path


SUPPORTED_SOURCE_MODELS = frozenset(("albef", "tcl", "clip_vit_b16", "clip_rn101"))


_SOURCE_BRIDGES = {
    "albef": ("vlpattack.legacy.models.albef_retrieval", "ALBEF",
              "936572ed7acc90e448bef95cd67731a933521c8925351ffb412252c74d37ee29"),
    "tcl": ("vlpattack.legacy.models.albef_retrieval", "ALBEF",
            "936572ed7acc90e448bef95cd67731a933521c8925351ffb412252c74d37ee29"),
    "clip_vit_b16": ("vlpattack.legacy.models.clip_model.model", "CLIP",
                     "11d30e2fb2662ed01ba3a516bd4f5d92a9a97ff5b0658019d9af447dfef796bd"),
    "clip_rn101": ("vlpattack.legacy.models.clip_model.model", "CLIP",
                   "11d30e2fb2662ed01ba3a516bd4f5d92a9a97ff5b0658019d9af447dfef796bd"),
}


class FidelityError(RuntimeError):
    """The requested execution cannot preserve the declared baseline contract."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _code_shape(code):
    """Compare executable code without binding it to a machine-specific path."""
    constants = [
        {"code": _code_shape(item)} if isinstance(item, types.CodeType)
        else {"literal": marshal.dumps(item).hex()}
        for item in code.co_consts
    ]
    return {
        "code": code.co_code.hex(), "consts": constants,
        "names": code.co_names, "vars": code.co_varnames,
        "free": code.co_freevars, "cell": code.co_cellvars,
        "args": code.co_argcount, "posonly": code.co_posonlyargcount,
        "kwonly": code.co_kwonlyargcount, "flags": code.co_flags,
        "exception_table": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _code_hash(code):
    return _sha(json.dumps(_code_shape(code), sort_keys=True).encode("utf-8"))


def _find_class_code(module_code, name):
    return next(value for value in module_code.co_consts
                if isinstance(value, types.CodeType) and value.co_name == name)


def verify_native_source_bridge(model_name, model, tokenizer):
    """Check the B1 loader's actual source bridge without replacing encoding.

    The caller may observe inference methods on the model instance for real
    operation counts; the underlying class and its pinned inference code must
    remain the original source. No proxy model, pooling, tokenization or native
    CLIP context is introduced by this check.
    """
    name = model_name
    if type(name) is not str or name not in SUPPORTED_SOURCE_MODELS:
        raise FidelityError("Unsupported original source family")
    module_name, class_name, source_sha = _SOURCE_BRIDGES[name]
    cls = type(model)
    if cls.__module__ != module_name or cls.__name__ != class_name:
        raise FidelityError("The actual source is not its original pinned B1 class")
    source_path = Path(inspect.getsourcefile(cls) or "")
    if not source_path.is_file() or _sha(source_path.read_bytes()) != source_sha:
        raise FidelityError("The original source-model bridge bytes differ")
    module_code = compile(source_path.read_bytes(), str(source_path), "exec", dont_inherit=True)
    class_code = _find_class_code(module_code, class_name)
    for method_name in ("inference_image", "inference_text"):
        actual = getattr(cls, method_name, None)
        expected = next(value for value in class_code.co_consts
                        if isinstance(value, types.CodeType) and value.co_name == method_name)
        if (not isinstance(actual, types.FunctionType)
                or _code_hash(actual.__code__) != _code_hash(expected)):
            raise FidelityError("The original source inference method differs: " + method_name)
    is_clip = name in {"clip_vit_b16", "clip_rn101"}
    if is_clip:
        if (getattr(model, "tokenizer", None) is not tokenizer
                or type(getattr(model, "context_length", None)) is not int
                or model.context_length != 77):
            raise FidelityError("Keep the original BERT-decode/native-CLIP77 text bridge")
        expected_visual = "VisionTransformer" if name == "clip_vit_b16" else "ModifiedResNet"
        visual_cls = type(model.visual)
        if visual_cls.__module__ != module_name or visual_cls.__name__ != expected_visual:
            raise FidelityError("The source label does not match its original CLIP visual family")
    return {
        "source_model": name, "class_module": module_name, "class_name": class_name,
        "source_bridge_sha256": source_sha,
        "attack_BERT_MLM_max_length": 30,
        "source_text_route": "original_BERT30_decode_then_native_CLIP77" if is_clip
                             else "original_ALBEF_TCL_BERT30_projected_CLS",
        "native_CLIP_context_length": 77 if is_clip else None,
        "source_pooling_and_tokenization_replaced": False,
        "inference_instance_observers_owned_by_caller": True,
        "per_source_server_numerical_acceptance": None,
    }
