from __future__ import annotations


import os


from pathlib import Path


def project_root() -> Path:
    """Return the repository root without depending on the current directory."""
    override = os.environ.get("VLPATTACK_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def ensure_within_project(path: Path) -> Path:
    resolved = path.resolve()
    root = project_root().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing path outside project: {resolved}")
    return resolved
