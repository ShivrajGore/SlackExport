from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingWriter:
    """Appends embedding-friendly text chunks to a JSONL file."""

    output_path: Path

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
