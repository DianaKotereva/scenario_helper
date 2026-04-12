"""Snapshot exporter for chapters/chunks/entities/relations/index."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    ActionByChapter,
    ChapterRecord,
    ChunkRecord,
    EntityChapterIndexRecord,
    EntityRecord,
    RelationRecord,
    SNAPSHOT_FILENAMES,
)

logger = logging.getLogger(__name__)


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        data = asdict(obj)
        return _to_jsonable(data)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def _iter_numeric_pickles(directory: Path, max_files: int = 0) -> List[Tuple[int, Path]]:
    if not directory.exists():
        return []
    records: List[Tuple[int, Path]] = []
    for fp in directory.glob("*.pkl"):
        stem = fp.stem
        if stem.isdigit():
            records.append((int(stem), fp))
    records.sort(key=lambda x: x[0])
    if max_files > 0:
        records = records[:max_files]
    return records


def _split_book_to_parts(book_path: Path) -> List[str]:
    text = book_path.read_text(encoding="utf-8")
    # Matches the preprocessing split separators: ["========== ", "***"]
    parts = [p.strip() for p in re.split(r"(?:========== |\*\*\*)", text) if p.strip()]
    return parts


def _build_chapters(book_path: Path, chapter_ids: Sequence[int]) -> List[ChapterRecord]:
    parts = _split_book_to_parts(book_path)
    chapters: List[ChapterRecord] = []
    for chapter_id in sorted(set(chapter_ids)):
        if chapter_id < 0:
            continue
        if chapter_id >= len(parts):
            logger.warning("Chapter %s is out of range for book split (%s parts)", chapter_id, len(parts))
            continue
        text = parts[chapter_id].strip()
        first_line = text.splitlines()[0].strip() if text else f"Chapter {chapter_id}"
        title = _normalize_whitespace(first_line)[:300]
        chapters.append(
            ChapterRecord(
                chapter_id=chapter_id,
                title=title,
                text=text,
                source_id=chapter_id,
            )
        )
    return chapters


def _chunk_chapter(chapter: ChapterRecord, chunk_size: int, chunk_overlap: int) -> List[ChunkRecord]:
    text = chapter.text
    if not text:
        return []
    chunks: List[ChunkRecord] = []
    start = 0
    idx = 0
    overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))
    while start < len(text):
        end = min(len(text), start + chunk_size)
        snippet = text[start:end].strip()
        if snippet:
            chunk_id = f"ch{chapter.chapter_id}_{idx}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=snippet,
                    source_id=chapter.source_id,
                    chapter_id=chapter.chapter_id,
                    start=start,
                    end=end,
                    metadata={"chunk_index": idx, "title": chapter.title},
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _build_chunks(chapters: Sequence[ChapterRecord], chunk_size: int, chunk_overlap: int) -> List[ChunkRecord]:
    rows: List[ChunkRecord] = []
    for chapter in chapters:
        rows.extend(_chunk_chapter(chapter, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return rows


def _normalize_actions_from_raw(raw_actions: Any, default_source_id: int) -> List[str]:
    if isinstance(raw_actions, list):
        normalized: List[str] = []
        for item in raw_actions:
            if isinstance(item, str):
                text = _normalize_whitespace(item)
                if text:
                    normalized.append(text)
                continue
            if isinstance(item, dict):
                sid = item.get("source_id", default_source_id)
                cid = item.get("chapter_id", sid)
                desc = _normalize_whitespace(str(item.get("description", "")))
                quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
                quotes_clean = [_normalize_whitespace(str(q)) for q in quotes if _normalize_whitespace(str(q))]
                if not desc:
                    continue
                if quotes_clean:
                    normalized.append(
                        f"[source_id={sid} chapter_id={cid}] {desc} | quotes: {' | '.join(quotes_clean)}"
                    )
                else:
                    normalized.append(f"[source_id={sid} chapter_id={cid}] {desc}")
        return normalized

    text = _normalize_whitespace(str(raw_actions))
    return [text] if text else []


def _normalize_relation_rows_from_raw(rel: Dict[str, Any], default_source_id: int) -> List[Dict[str, Any]]:
    source = str(rel.get("source_node_id", "")).strip()
    target = str(rel.get("target_node_id", "")).strip()
    if not source or not target:
        return []

    relation_type = str(rel.get("type", "")).strip()
    descriptions = rel.get("descriptions")
    if descriptions is None and rel.get("description") is not None:
        descriptions = rel.get("description")

    rows: List[Dict[str, Any]] = []
    if isinstance(descriptions, str):
        description = _normalize_whitespace(descriptions)
        rows.append(
            {
                "source": source,
                "target": target,
                "type": relation_type,
                "description": description,
                "chapter_id": default_source_id,
                "source_id": default_source_id,
            }
        )
        return rows

    if isinstance(descriptions, list):
        for item in descriptions:
            if isinstance(item, str):
                text = _normalize_whitespace(item)
                if text:
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "type": relation_type,
                            "description": text,
                            "chapter_id": default_source_id,
                            "source_id": default_source_id,
                        }
                    )
                continue
            if not isinstance(item, dict):
                continue
            sid = item.get("source_id", default_source_id)
            cid = item.get("chapter_id", sid)
            desc = _normalize_whitespace(str(item.get("description", "")))
            quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
            quotes_clean = [_normalize_whitespace(str(q)) for q in quotes if _normalize_whitespace(str(q))]
            if not desc:
                continue
            if quotes_clean:
                desc = f"{desc} | quotes: {' | '.join(quotes_clean)}"
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "description": desc,
                    "chapter_id": int(cid) if isinstance(cid, int) else default_source_id,
                    "source_id": int(sid) if isinstance(sid, int) else default_source_id,
                }
            )
        return rows

    return rows


def _extract_raw_from_results(results_dir: Path, max_files: int) -> Tuple[List[EntityRecord], List[RelationRecord], List[int]]:
    entities: List[EntityRecord] = []
    relations: List[RelationRecord] = []
    chapter_ids: List[int] = []

    def _coerce_payload(raw: Any, chapter_id: int, fp: Path) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        logger.warning(
            "Skip non-dict extraction payload for chapter %s from %s (type=%s)",
            chapter_id,
            fp,
            type(raw).__name__,
        )
        return None

    for chapter_id, fp in _iter_numeric_pickles(results_dir, max_files=max_files):
        data_raw = pickle.load(fp.open("rb"))
        data = _coerce_payload(data_raw, chapter_id, fp)
        if not data:
            continue
        chapter_ids.append(chapter_id)
        for node in data.get("nodes", []):
            main_name = str(node.get("main_name", "")).strip()
            if not main_name:
                continue
            alt_names = [str(v).strip() for v in node.get("alt_names", []) if str(v).strip()]
            actions = _normalize_actions_from_raw(node.get("actions"), chapter_id)
            payload = {"chapter_id": chapter_id, "main_name": main_name, "classification": node.get("classification", "")}
            entity_id = _stable_id("raw_ent", payload)
            entities.append(
                EntityRecord(
                    entity_id=entity_id,
                    main_name=main_name,
                    alt_names=alt_names,
                    classification=str(node.get("classification", "")),
                    actions=actions,
                    chapter_id=chapter_id,
                    source_id=chapter_id,
                    record_type="raw",
                    metadata={"origin": "results"},
                )
            )

        for rel in data.get("relations", []):
            for rel_row in _normalize_relation_rows_from_raw(rel, chapter_id):
                payload = {
                    "chapter_id": rel_row["chapter_id"],
                    "source": rel_row["source"],
                    "target": rel_row["target"],
                    "type": rel_row["type"],
                    "description": rel_row["description"],
                }
                relations.append(
                    RelationRecord(
                        relation_id=_stable_id("raw_rel", payload),
                        source_entity=rel_row["source"],
                        target_entity=rel_row["target"],
                        type=rel_row["type"],
                        description=rel_row["description"],
                        chapter_id=rel_row["chapter_id"],
                        source_id=rel_row["source_id"],
                        record_type="raw",
                        metadata={"origin": "results"},
                    )
                )
    return entities, relations, chapter_ids


def _extract_raw_from_graph_pickles(
    graph_nodes_dir: Path, graph_relations_dir: Path, max_files: int
) -> Tuple[List[EntityRecord], List[RelationRecord], List[int]]:
    entities: List[EntityRecord] = []
    relations: List[RelationRecord] = []
    chapter_ids: List[int] = []

    node_files = _iter_numeric_pickles(graph_nodes_dir, max_files=max_files)
    rel_files = {sid: fp for sid, fp in _iter_numeric_pickles(graph_relations_dir, max_files=max_files)}

    for chapter_id, node_fp in node_files:
        chapter_ids.append(chapter_id)
        chapter_nodes = pickle.load(node_fp.open("rb"))
        for node in chapter_nodes.nodes.values():
            actions = [_normalize_whitespace(a.action) for a in getattr(node, "actions", []) if _normalize_whitespace(a.action)]
            payload = {"chapter_id": chapter_id, "main_name": node.main_name, "classification": node.classification}
            entities.append(
                EntityRecord(
                    entity_id=_stable_id("raw_ent", payload),
                    main_name=node.main_name,
                    alt_names=list(node.alt_names or []),
                    classification=node.classification,
                    actions=actions,
                    chapter_id=chapter_id,
                    source_id=chapter_id,
                    record_type="raw",
                    metadata={"origin": "graph_nodes"},
                )
            )

        rel_fp = rel_files.get(chapter_id)
        if not rel_fp:
            continue
        chapter_relations = pickle.load(rel_fp.open("rb"))
        for edge in chapter_relations.relationships.values():
            source = edge.object_1
            target = edge.object_2
            for desc in edge.description:
                source_ids = [int(v) for v in (desc.source_id or ()) if isinstance(v, int)]
                chapter_ref = source_ids[0] if source_ids else chapter_id
                payload = {
                    "chapter_id": chapter_ref,
                    "source": source,
                    "target": target,
                    "type": desc.type,
                    "description": desc.description,
                }
                relations.append(
                    RelationRecord(
                        relation_id=_stable_id("raw_rel", payload),
                        source_entity=source,
                        target_entity=target,
                        type=desc.type,
                        description=_normalize_whitespace(desc.description),
                        chapter_id=chapter_ref,
                        source_id=chapter_ref,
                        record_type="raw",
                        metadata={"origin": "graph_relations", "source_ids": source_ids},
                    )
                )

    return entities, relations, chapter_ids


def _extract_merged_entities_and_relations(
    merged_graph_path: Optional[Path],
) -> Tuple[List[EntityRecord], List[RelationRecord], List[EntityChapterIndexRecord]]:
    if not merged_graph_path or not merged_graph_path.exists():
        return [], [], []

    merged_graph = pickle.load(merged_graph_path.open("rb"))
    entities: List[EntityRecord] = []
    relations: List[RelationRecord] = []
    entity_index: List[EntityChapterIndexRecord] = []

    for node in merged_graph.nodes.nodes.values():
        per_chapter: Dict[int, List[str]] = {}
        for action in node.actions or []:
            action_text = _normalize_whitespace(getattr(action, "action", ""))
            if not action_text:
                continue
            source_ids = [int(v) for v in (getattr(action, "source_id", ()) or ()) if isinstance(v, int)]
            if not source_ids:
                continue
            for chapter_id in source_ids:
                per_chapter.setdefault(chapter_id, []).append(action_text)

        chapter_ids = sorted(per_chapter.keys())
        actions_by_chapter = [
            ActionByChapter(
                chapter_id=chapter_id,
                summary=_normalize_whitespace(" ".join(per_chapter[chapter_id])),
                evidence_refs=[{"source_id": chapter_id}],
            )
            for chapter_id in chapter_ids
        ]
        mentions_count = sum(len(v) for v in per_chapter.values())
        entity_payload = {
            "main_name": node.main_name,
            "classification": node.classification,
            "chapter_ids": chapter_ids,
            "alt_names": sorted(node.alt_names or []),
        }
        entity_id = _stable_id("merged_ent", entity_payload)
        entities.append(
            EntityRecord(
                entity_id=entity_id,
                main_name=node.main_name,
                alt_names=list(node.alt_names or []),
                classification=node.classification,
                chapter_ids=chapter_ids,
                source_ids=list(chapter_ids),
                actions_by_chapter=actions_by_chapter,
                record_type="merged",
                metadata={"mentions_count": mentions_count, "origin": "merged_graph"},
            )
        )
        entity_index.append(
            EntityChapterIndexRecord(
                entity_id=entity_id,
                main_name=node.main_name,
                chapters=chapter_ids,
                aliases=list(node.alt_names or []),
                mentions_count=mentions_count,
                metadata={"origin": "merged_graph"},
            )
        )

    for edge in merged_graph.relationships.relationships.values():
        source = edge.object_1
        target = edge.object_2
        for desc in edge.description:
            source_ids = [int(v) for v in (desc.source_id or ()) if isinstance(v, int)]
            chapter_ref = source_ids[0] if source_ids else None
            payload = {
                "source": source,
                "target": target,
                "type": desc.type,
                "description": desc.description,
                "source_ids": source_ids,
            }
            relations.append(
                RelationRecord(
                    relation_id=_stable_id("merged_rel", payload),
                    source_entity=source,
                    target_entity=target,
                    type=desc.type,
                    description=_normalize_whitespace(desc.description),
                    chapter_id=chapter_ref,
                    source_id=chapter_ref,
                    record_type="merged",
                    metadata={"source_ids": source_ids, "origin": "merged_graph"},
                )
            )

    return entities, relations, entity_index


def export_snapshot(
    *,
    output_dir: Path,
    book_path: Path,
    max_files: int = 0,
    results_dir: Optional[Path] = None,
    graph_nodes_dir: Optional[Path] = None,
    graph_relations_dir: Optional[Path] = None,
    merged_graph_path: Optional[Path] = None,
    chunk_size: int = 1400,
    chunk_overlap: int = 250,
) -> Dict[str, Any]:
    """Export snapshot JSONL files from preprocessing artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_entities: List[EntityRecord] = []
    raw_relations: List[RelationRecord] = []
    chapter_ids: List[int] = []

    if results_dir and results_dir.exists():
        raw_entities, raw_relations, chapter_ids = _extract_raw_from_results(results_dir, max_files=max_files)
        if raw_entities:
            logger.info("Raw snapshot extracted from results: entities=%s relations=%s", len(raw_entities), len(raw_relations))

    if not raw_entities and graph_nodes_dir and graph_relations_dir:
        raw_entities, raw_relations, chapter_ids = _extract_raw_from_graph_pickles(
            graph_nodes_dir=graph_nodes_dir,
            graph_relations_dir=graph_relations_dir,
            max_files=max_files,
        )
        logger.info(
            "Raw snapshot extracted from graph pickles: entities=%s relations=%s",
            len(raw_entities),
            len(raw_relations),
        )

    merged_entities, merged_relations, entity_index = _extract_merged_entities_and_relations(merged_graph_path)
    if not chapter_ids:
        inferred_ids: set[int] = set()
        for item in merged_entities:
            for cid in item.chapter_ids or []:
                if isinstance(cid, int):
                    inferred_ids.add(cid)
        for item in merged_relations:
            if isinstance(item.chapter_id, int):
                inferred_ids.add(item.chapter_id)
            meta_source_ids = (item.metadata or {}).get("source_ids", [])
            if isinstance(meta_source_ids, list):
                inferred_ids.update(int(v) for v in meta_source_ids if isinstance(v, int))
        chapter_ids = sorted(inferred_ids)

    if not chapter_ids:
        raise RuntimeError(
            "No chapter ids found for snapshot export. "
            "Provide results_dir/graph_* dirs or merged_graph with source_id evidence."
        )

    chapters = _build_chapters(book_path=book_path, chapter_ids=chapter_ids)
    chunks = _build_chunks(chapters=chapters, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    entities_rows = [asdict(v) for v in raw_entities] + [asdict(v) for v in merged_entities]
    relations_rows = [asdict(v) for v in raw_relations] + [asdict(v) for v in merged_relations]

    counts = {
        "chapters": _write_jsonl(output_dir / SNAPSHOT_FILENAMES["chapters"], [asdict(v) for v in chapters]),
        "chunks": _write_jsonl(output_dir / SNAPSHOT_FILENAMES["chunks"], [asdict(v) for v in chunks]),
        "entities": _write_jsonl(output_dir / SNAPSHOT_FILENAMES["entities"], entities_rows),
        "relations": _write_jsonl(output_dir / SNAPSHOT_FILENAMES["relations"], relations_rows),
        "entity_chapter_index": _write_jsonl(
            output_dir / SNAPSHOT_FILENAMES["entity_chapter_index"],
            [asdict(v) for v in entity_index],
        ),
    }

    return {
        "output_dir": str(output_dir),
        "counts": counts,
        "chapter_ids": sorted(set(chapter_ids)),
        "raw_entities": len(raw_entities),
        "merged_entities": len(merged_entities),
        "raw_relations": len(raw_relations),
        "merged_relations": len(merged_relations),
    }
