from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .io_integrity import exclusive_file_lock, write_text_atomic


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    dataset: str
    attack: str
    source_model: str
    target_model: str
    direction: str
    metric: str
    k: int
    value: float
    seed: int
    experiment: str = ""
    variant: str = ""
    defense: str = "none"
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if not value["created_at"]:
            value["created_at"] = datetime.now(timezone.utc).isoformat()
        return value


def append_records(path: str | Path, records: Iterable[MetricRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    if not lines:
        return
    with exclusive_file_lock(output.with_name(f".{output.name}.lock")):
        existing = output.read_text(encoding="utf-8") if output.is_file() else ""
        write_text_atomic(output, existing + "".join(f"{line}\n" for line in lines))


def write_records_atomic(path: str | Path, records: Iterable[MetricRecord]) -> None:
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    write_text_atomic(Path(path), "".join(f"{line}\n" for line in lines))


def read_records(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {error}") from error
    return rows
