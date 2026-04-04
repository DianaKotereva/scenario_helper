"""Validation for exported snapshot artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .schema import REQUIRED_FIELDS, SNAPSHOT_FILENAMES


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid json: {exc}") from exc
    return rows


def _normalized_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _missing_required_fields(rows: Iterable[Dict[str, Any]], required: List[str]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            errors.append({"row_index": idx, "missing_fields": missing})
    return errors


def validate_snapshot(snapshot_dir: Path) -> Dict[str, Any]:
    """Validate snapshot files and merge-level consistency."""

    chapters = _read_jsonl(snapshot_dir / SNAPSHOT_FILENAMES["chapters"])
    chunks = _read_jsonl(snapshot_dir / SNAPSHOT_FILENAMES["chunks"])
    entities = _read_jsonl(snapshot_dir / SNAPSHOT_FILENAMES["entities"])
    relations = _read_jsonl(snapshot_dir / SNAPSHOT_FILENAMES["relations"])
    entity_index = _read_jsonl(snapshot_dir / SNAPSHOT_FILENAMES["entity_chapter_index"])

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for key, rows in [
        ("chapters", chapters),
        ("chunks", chunks),
        ("entities", entities),
        ("relations", relations),
        ("entity_chapter_index", entity_index),
    ]:
        missing = _missing_required_fields(rows, REQUIRED_FIELDS[key])
        if missing:
            errors.append({"type": "missing_fields", "file": key, "details": missing[:20], "count": len(missing)})

    chapter_ids = {int(row["chapter_id"]) for row in chapters if isinstance(row.get("chapter_id"), int)}
    merged_entities = [row for row in entities if row.get("record_type") == "merged"]
    raw_entities = [row for row in entities if row.get("record_type") == "raw"]
    merged_relations = [row for row in relations if row.get("record_type") == "merged"]
    raw_relations = [row for row in relations if row.get("record_type") == "raw"]

    # Core linkage checks.
    bad_chunks = [
        row
        for row in chunks
        if not isinstance(row.get("chapter_id"), int)
        or not isinstance(row.get("source_id"), int)
        or row.get("chapter_id") not in chapter_ids
    ]
    if bad_chunks:
        errors.append({"type": "chunk_linkage", "count": len(bad_chunks), "sample": bad_chunks[:10]})

    bad_raw_entities = [
        row
        for row in raw_entities
        if not isinstance(row.get("chapter_id"), int) or row.get("chapter_id") not in chapter_ids
    ]
    if bad_raw_entities:
        errors.append({"type": "raw_entity_linkage", "count": len(bad_raw_entities), "sample": bad_raw_entities[:10]})

    bad_raw_relations = [
        row
        for row in raw_relations
        if not isinstance(row.get("chapter_id"), int) or row.get("chapter_id") not in chapter_ids
    ]
    if bad_raw_relations:
        errors.append({"type": "raw_relation_linkage", "count": len(bad_raw_relations), "sample": bad_raw_relations[:10]})

    # Merge-level checks.
    merged_without_chapters = [
        row for row in merged_entities if not row.get("chapter_ids") or not isinstance(row.get("chapter_ids"), list)
    ]
    if merged_without_chapters:
        errors.append(
            {
                "type": "merge_without_chapters",
                "count": len(merged_without_chapters),
                "sample_entity_ids": [row.get("entity_id") for row in merged_without_chapters[:20]],
            }
        )

    merged_entities_by_id = {row.get("entity_id"): row for row in merged_entities}
    merged_entities_by_name = {_normalized_name(str(row.get("main_name", ""))): row for row in merged_entities}

    # entity_chapter_index consistency with merged entities.
    bad_index_rows: List[Dict[str, Any]] = []
    for row in entity_index:
        entity_id = row.get("entity_id")
        merged = merged_entities_by_id.get(entity_id)
        if not merged:
            bad_index_rows.append({"reason": "unknown_entity_id", "row": row})
            continue
        expected_chapters = sorted(set(merged.get("chapter_ids") or []))
        got_chapters = sorted(set(row.get("chapters") or []))
        if expected_chapters != got_chapters:
            bad_index_rows.append(
                {
                    "reason": "chapter_mismatch",
                    "entity_id": entity_id,
                    "expected_chapters": expected_chapters,
                    "got_chapters": got_chapters,
                }
            )
    if bad_index_rows:
        errors.append({"type": "entity_chapter_index_consistency", "count": len(bad_index_rows), "details": bad_index_rows[:20]})

    # Merged relation endpoints must exist as merged entities.
    bad_merged_relations = []
    for row in merged_relations:
        src = _normalized_name(str(row.get("source_entity", "")))
        dst = _normalized_name(str(row.get("target_entity", "")))
        if src not in merged_entities_by_name or dst not in merged_entities_by_name:
            bad_merged_relations.append(row)
    if bad_merged_relations:
        errors.append(
            {
                "type": "merged_relation_orphans",
                "count": len(bad_merged_relations),
                "sample": bad_merged_relations[:20],
            }
        )

    # Alias collisions (under-merge signal).
    alias_to_entities: Dict[str, set] = {}
    for row in merged_entities:
        entity_id = row.get("entity_id")
        aliases = [row.get("main_name", "")] + list(row.get("alt_names") or [])
        for alias in aliases:
            norm_alias = _normalized_name(str(alias))
            if len(norm_alias) < 3:
                continue
            alias_to_entities.setdefault(norm_alias, set()).add(entity_id)

    collisions = [
        {"alias": alias, "entity_ids": sorted(list(entity_ids))}
        for alias, entity_ids in alias_to_entities.items()
        if len(entity_ids) > 1
    ]
    if collisions:
        warnings.append({"type": "alias_collisions", "count": len(collisions), "sample": collisions[:30]})

    report = {
        "status": "ok" if not errors else "fail",
        "counts": {
            "chapters": len(chapters),
            "chunks": len(chunks),
            "entities_total": len(entities),
            "entities_raw": len(raw_entities),
            "entities_merged": len(merged_entities),
            "relations_total": len(relations),
            "relations_raw": len(raw_relations),
            "relations_merged": len(merged_relations),
            "entity_chapter_index": len(entity_index),
        },
        "merge_checks": {
            "merged_without_chapters": len(merged_without_chapters),
            "merged_relation_orphans": len([e for e in errors if e["type"] == "merged_relation_orphans"]),
            "alias_collision_count": len(collisions),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report
