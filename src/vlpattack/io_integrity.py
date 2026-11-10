from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Cannot fsync a missing, non-file, or symlink path: {path}")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.part")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _try_lock(handle) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(
    path: Path,
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.2,
) -> Iterator[None]:
    """Hold one cross-process advisory lock without deleting its inode.

    The lock file remains as non-authoritative diagnostic metadata. Lock
    ownership is the operating-system lock, not the JSON stored in the file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        started = time.monotonic()
        while not _try_lock(handle):
            if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
                raise RuntimeError(f"Timed out acquiring lock: {path}")
            time.sleep(poll_seconds)

        try:
            metadata = {
                "schema_version": 1,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_unix_ns": time.time_ns(),
            }
            encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
            handle.seek(0)
            handle.write(encoded)
            handle.truncate(handle.tell())
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            _unlock(handle)
