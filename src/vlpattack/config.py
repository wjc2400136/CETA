from __future__ import annotations


from pathlib import Path


from typing import Any


from ruamel.yaml import YAML


from .paths import project_root, resolve_project_path


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    if not resolved.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved}")
    yaml = YAML(typ="safe")
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a mapping in {resolved}")
    return dict(value)


def named_config(kind: str, name: str) -> dict[str, Any]:
    safe_kind = kind.replace("-", "_")
    safe_name = name.lower().replace("-", "_")
    path = project_root() / "configs" / safe_kind / f"{safe_name}.yaml"
    return load_yaml(path)
