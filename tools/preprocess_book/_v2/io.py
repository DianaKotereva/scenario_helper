from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tools.preprocess_book.storage.storage import FileManager


def parse_source_id(filename: str) -> Tuple[int, ...]:
    source_id_str = filename.replace(".pkl", "")
    try:
        source_id = ast.literal_eval(source_id_str)
        if isinstance(source_id, tuple):
            return source_id
        return (int(source_id),)
    except (ValueError, SyntaxError, TypeError):
        parts = source_id_str.split("_")
        source_id = tuple(int(p) for p in parts if p.isdigit())
        if not source_id:
            raise ValueError(f"Failed to parse source_id from {filename}")
        return source_id


def sorted_result_files(results_dir: Path, max_files: int | None = None) -> List[Path]:
    files = sorted(
        [p for p in results_dir.glob("*.pkl")],
        key=lambda p: parse_source_id(p.name)[0],
    )
    if max_files and max_files > 0:
        return files[:max_files]
    return files


def load_results(results_dir: Path, max_files: int | None = None) -> List[dict]:
    items: List[dict] = []
    for path in sorted_result_files(results_dir, max_files=max_files):
        payload = FileManager.load_pickle(path)
        if not isinstance(payload, dict):
            continue
        payload["_source_id"] = parse_source_id(path.name)
        payload["_filename"] = path.name
        items.append(payload)
    return items


def make_run_prefix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

