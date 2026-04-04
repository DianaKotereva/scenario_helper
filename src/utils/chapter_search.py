from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import settings

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {t for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower()) if len(t) >= 3}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


@lru_cache(maxsize=2)
def load_snapshot(snapshot_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    base = Path(snapshot_dir)
    data = {
        "entities_merged": _read_jsonl(base / "entities_merged.jsonl"),
        "relations_merged": _read_jsonl(base / "relations_merged.jsonl"),
        "chapters": _read_jsonl(base / "chapters.jsonl"),
    }
    logger.info(
        "Loaded snapshot from %s: entities=%s relations=%s chapters=%s",
        base,
        len(data["entities_merged"]),
        len(data["relations_merged"]),
        len(data["chapters"]),
    )
    return data


def _entity_text(ent: Dict[str, Any]) -> str:
    aliases = " ".join(ent.get("alt_names") or [])
    actions = " ".join(item.get("summary", "") for item in (ent.get("actions_by_chapter") or []))
    return f"{ent.get('main_name', '')} {aliases} {actions}"


def _relation_text(rel: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(rel.get("source_entity", "")),
            str(rel.get("target_entity", "")),
            str(rel.get("type", "")),
            str(rel.get("description", "")),
        ]
    )


def guided_deterministic_search(
    query: str,
    input_hints: Optional[Dict[str, Any]] = None,
    snapshot_dir: Optional[str] = None,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
    max_chapters: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Фаза B: детерминированный поиск по snapshot (entities/relations/chapters),
    приоритезируя навигационные hints из фазы A.
    """
    hints = input_hints or {}
    query_tokens = _tokenize(query)
    snapshot = load_snapshot(snapshot_dir or settings.SNAPSHOT_DIR)

    max_entities = max_entities or settings.PHASE_B_MAX_ENTITIES
    max_relations = max_relations or settings.PHASE_B_MAX_RELATIONS
    max_chapters = max_chapters or settings.PHASE_B_MAX_CHAPTERS

    hint_chapter_ids = {int(x) for x in (hints.get("chapter_ids") or []) if isinstance(x, int)}
    hint_entity_ids = {str(x) for x in (hints.get("entity_ids") or []) if isinstance(x, str)}
    hint_entity_tokens = {str(x).lower() for x in (hints.get("entity_name_tokens") or []) if isinstance(x, str)}

    entity_scored: List[Dict[str, Any]] = []
    entity_rejected: List[Dict[str, Any]] = []
    selected_entity_ids: Set[str] = set()
    selected_chapter_ids: Set[int] = set(hint_chapter_ids)

    for ent in snapshot["entities_merged"]:
        eid = str(ent.get("entity_id", ""))
        if not eid:
            continue
        ent_tokens = _tokenize(_entity_text(ent))
        overlap = query_tokens.intersection(ent_tokens)
        score = float(len(overlap))
        if eid in hint_entity_ids:
            score += 2.5
        if query_tokens.intersection(hint_entity_tokens.intersection(ent_tokens)):
            score += 1.5
        if score <= 0:
            entity_rejected.append({"entity_id": eid, "reason": "no_overlap"})
            continue
        entity_scored.append(
            {
                "entity_id": eid,
                "main_name": ent.get("main_name", ""),
                "score": score,
                "chapters": ent.get("chapter_ids") or [],
                "summary": " | ".join(
                    (row.get("summary", "")[:180] for row in (ent.get("actions_by_chapter") or [])[:2])
                ),
            }
        )

    entity_scored.sort(key=lambda x: x["score"], reverse=True)
    selected_entities = entity_scored[:max_entities]
    for item in selected_entities:
        selected_entity_ids.add(item["entity_id"])
        for cid in item.get("chapters", []):
            if isinstance(cid, int):
                selected_chapter_ids.add(cid)

    relation_scored: List[Dict[str, Any]] = []
    relation_rejected: List[Dict[str, Any]] = []
    for rel in snapshot["relations_merged"]:
        src = str(rel.get("source_entity", ""))
        dst = str(rel.get("target_entity", ""))
        text_tokens = _tokenize(_relation_text(rel))
        overlap = query_tokens.intersection(text_tokens)

        entity_hit = False
        if selected_entities:
            selected_names = {e["main_name"].lower() for e in selected_entities if e.get("main_name")}
            entity_hit = src.lower() in selected_names or dst.lower() in selected_names

        score = float(len(overlap)) + (2.0 if entity_hit else 0.0)
        if score <= 0:
            relation_rejected.append({"relation_id": rel.get("relation_id"), "reason": "no_overlap"})
            continue

        relation_scored.append(
            {
                "relation_id": rel.get("relation_id"),
                "source_entity": src,
                "target_entity": dst,
                "type": rel.get("type", ""),
                "score": score,
                "description": str(rel.get("description", ""))[:220],
                "chapter_id": rel.get("chapter_id"),
                "source_id": rel.get("source_id"),
            }
        )

    relation_scored.sort(key=lambda x: x["score"], reverse=True)
    selected_relations = relation_scored[:max_relations]
    for rel in selected_relations:
        cid = rel.get("chapter_id")
        if not isinstance(cid, int):
            sid = rel.get("source_id")
            if isinstance(sid, int):
                cid = sid
                rel["chapter_id"] = sid
        if isinstance(cid, int):
            selected_chapter_ids.add(cid)

    chapters_by_id = {
        int(ch["chapter_id"]): ch for ch in snapshot["chapters"] if isinstance(ch.get("chapter_id"), int)
    }
    selected_chapters: List[Dict[str, Any]] = []
    for cid in sorted(selected_chapter_ids):
        if cid not in chapters_by_id:
            continue
        ch = chapters_by_id[cid]
        selected_chapters.append(
            {
                "chapter_id": cid,
                "source_id": cid,
                "title": ch.get("title", ""),
                "snippet": str(ch.get("text", ""))[:600],
            }
        )
        if len(selected_chapters) >= max_chapters:
            break

    return {
        "entity_ids": [e["entity_id"] for e in selected_entities],
        "chapter_ids": [c["chapter_id"] for c in selected_chapters],
        "entities": selected_entities,
        "relations": selected_relations,
        "chapters": selected_chapters,
        "selected_items": {
            "entities": selected_entities,
            "relations": selected_relations,
            "chapters": selected_chapters,
        },
        "rejected_items": {
            "entities": entity_rejected[:30],
            "relations": relation_rejected[:30],
        },
        "reason": "guided deterministic search over snapshot",
    }
