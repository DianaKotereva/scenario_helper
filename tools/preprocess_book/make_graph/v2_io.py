from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


def _parse_batch_file_range(path: Path) -> Tuple[int, int] | None:
    match = re.match(r"batch_\d+_(\d+)_(\d+)\.json$", path.name)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        start, end = end, start
    return start, end


def _iter_batch_files(results_dir: Path) -> List[Path]:
    batch_dir = results_dir / "_batch_payloads"
    if not batch_dir.exists():
        return []
    files = [p for p in batch_dir.glob("batch_*.json") if _parse_batch_file_range(p) is not None]
    files.sort(key=lambda p: _parse_batch_file_range(p)[0])  # type: ignore[index]
    return files


def _filter_payload_to_source_ids(payload: Dict[str, Any], allowed: set[int]) -> Dict[str, Any]:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []

    filtered_relations: List[Dict[str, Any]] = []
    endpoint_names: set[str] = set()
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        descriptions = rel.get("descriptions") if isinstance(rel.get("descriptions"), list) else []
        filtered_desc = [
            d
            for d in descriptions
            if isinstance(d, dict) and isinstance(d.get("source_id"), int) and int(d["source_id"]) in allowed
        ]
        if not filtered_desc:
            continue

        rel_copy = dict(rel)
        rel_copy["descriptions"] = filtered_desc
        filtered_relations.append(rel_copy)

        source_name = str(rel_copy.get("source_node_id", "")).strip()
        target_name = str(rel_copy.get("target_node_id", "")).strip()
        if source_name:
            endpoint_names.add(source_name)
        if target_name:
            endpoint_names.add(target_name)

    filtered_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        actions = node.get("actions") if isinstance(node.get("actions"), list) else []
        filtered_actions = [
            a
            for a in actions
            if isinstance(a, dict) and isinstance(a.get("source_id"), int) and int(a["source_id"]) in allowed
        ]
        main_name = str(node.get("main_name", "")).strip()
        keep = bool(filtered_actions) or (main_name in endpoint_names)
        if not keep:
            continue
        node_copy = dict(node)
        node_copy["actions"] = filtered_actions
        filtered_nodes.append(node_copy)

    return {
        "nodes": filtered_nodes,
        "relations": filtered_relations,
        "summarization": str(payload.get("summarization", "") or ""),
    }


def _load_results_from_batch_payloads(results_dir: Path, max_files: int | None = None) -> List[dict]:
    items: List[dict] = []
    batch_files = _iter_batch_files(results_dir)
    if not batch_files:
        return items

    max_source_id = (max_files - 1) if (max_files and max_files > 0) else None

    for path in batch_files:
        span = _parse_batch_file_range(path)
        if span is None:
            continue
        start, end = span
        source_ids = tuple(range(start, end + 1))
        if max_source_id is not None:
            allowed = {sid for sid in source_ids if sid <= max_source_id}
            if not allowed:
                continue
        else:
            allowed = set(source_ids)

        try:
            payload_raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload_raw, dict):
            continue

        payload = _filter_payload_to_source_ids(payload_raw, allowed)
        if not payload.get("nodes") and not payload.get("relations"):
            continue
        payload["_source_id"] = tuple(sorted(allowed))
        payload["_filename"] = path.name
        items.append(payload)

    return items


def load_results(
    results_dir: Path,
    max_files: int | None = None,
    prefer_batch_payloads: bool = True,
) -> List[dict]:
    if prefer_batch_payloads:
        batch_items = _load_results_from_batch_payloads(results_dir, max_files=max_files)
        if batch_items:
            return batch_items

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

