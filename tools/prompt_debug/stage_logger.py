from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CaseMeta:
    case_id: str
    row_id: int
    question: str
    expected: str


class StageLogger:
    """Collects stage-level diagnostic events and writes JSONL artifacts."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self._events_by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._cases: Dict[str, CaseMeta] = {}

    def register_case(self, meta: CaseMeta) -> None:
        self._cases[meta.case_id] = meta

    def log(self, case_id: str, stage: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts": datetime.now().isoformat(),
            "run_id": self.run_id,
            "case_id": case_id,
            "stage": stage,
            "payload": payload,
        }
        self._events_by_case[case_id].append(event)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @property
    def events_by_case(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._events_by_case

    @property
    def cases(self) -> Dict[str, CaseMeta]:
        return self._cases
