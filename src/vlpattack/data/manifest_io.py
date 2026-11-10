from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..paths import project_root


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".part",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            try:
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                pass
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
