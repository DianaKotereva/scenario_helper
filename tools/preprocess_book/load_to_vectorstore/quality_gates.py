"""
Quality gates for preprocess pipeline.

These checks are fail-fast: any mismatch raises ValueError.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from langchain_core.documents import Document

from src.utils.graph_search import BookGraph


REQUIRED_META_COMMON = ("source", "source_id")
REQUIRED_META_BY_SOURCE = {
    "nodes": ("schema_version", "graph_doc_type", "entity_name", "fact_index", "doc_uid"),
    "relations": (
        "schema_version",
        "graph_doc_type",
        "relation_object_1",
        "relation_object_2",
        "relation_type",
        "fact_index",
        "doc_uid",
    ),
    "summary": tuple(),
    "book": tuple(),
}


def _stable_fallback_uid(doc: Document) -> str:
    meta = doc.metadata or {}
    source = str(meta.get("source", "unknown"))
    source_id = str(meta.get("source_id", "unknown"))
    payload = f"{source}|{source_id}|{doc.page_content.strip()}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{source}:{source_id}:{digest}"


def doc_uid(doc: Document) -> str:
    uid = doc.metadata.get("doc_uid") if doc.metadata else None
    if uid:
        return str(uid)
    return _stable_fallback_uid(doc)


def expected_graph_fact_counts(book_graph: BookGraph) -> Tuple[int, int]:
    node_facts = 0
    relation_facts = 0

    for node in book_graph.nodes.nodes.values():
        for action in getattr(node, "actions", []) or []:
            if getattr(action, "action", None):
                node_facts += 1

    for relation in book_graph.relationships.relationships.values():
        for desc in getattr(relation, "description", []) or []:
            if getattr(desc, "description", None):
                relation_facts += 1

    return node_facts, relation_facts


def _validate_metadata_shape(documents: Iterable[Document]) -> None:
    for idx, doc in enumerate(documents):
        meta = doc.metadata or {}
        missing_common = [k for k in REQUIRED_META_COMMON if k not in meta]
        if missing_common:
            raise ValueError(f"Doc[{idx}] missing common metadata keys: {missing_common}")

        if not isinstance(meta.get("source"), str):
            raise ValueError(f"Doc[{idx}] metadata.source must be str")
        if not isinstance(meta.get("source_id"), int):
            raise ValueError(f"Doc[{idx}] metadata.source_id must be int")

        source = meta["source"]
        required = REQUIRED_META_BY_SOURCE.get(source, tuple())
        missing_required = [k for k in required if k not in meta]
        if missing_required:
            raise ValueError(
                f"Doc[{idx}] source='{source}' missing required metadata keys: {missing_required}"
            )

        text = (doc.page_content or "").strip()
        if not text:
            raise ValueError(f"Doc[{idx}] has empty page_content")


def _validate_duplicates(documents: Iterable[Document]) -> None:
    uids = [doc_uid(d) for d in documents]
    duplicates = [uid for uid, cnt in Counter(uids).items() if cnt > 1]
    if duplicates:
        preview = duplicates[:10]
        raise ValueError(
            "Duplicate documents detected by doc_uid. "
            f"count={len(duplicates)}, examples={preview}"
        )


def validate_graph_documents(
    documents: List[Document],
    expected_node_facts: int,
    expected_relation_facts: int,
) -> Dict[str, int]:
    _validate_metadata_shape(documents)
    _validate_duplicates(documents)

    by_source = Counter(d.metadata.get("source") for d in documents)
    node_docs = by_source.get("nodes", 0)
    relation_docs = by_source.get("relations", 0)

    if node_docs != expected_node_facts:
        raise ValueError(
            f"Graph node facts mismatch: expected={expected_node_facts}, actual={node_docs}"
        )
    if relation_docs != expected_relation_facts:
        raise ValueError(
            "Graph relation facts mismatch: "
            f"expected={expected_relation_facts}, actual={relation_docs}"
        )

    return {
        "total_docs": len(documents),
        "nodes_docs": node_docs,
        "relations_docs": relation_docs,
    }


def validate_chapters(chapters: List[Document]) -> Dict[str, int]:
    if not chapters:
        raise ValueError("No chapters found for indexing")

    _validate_metadata_shape(chapters)

    source_ids = [d.metadata["source_id"] for d in chapters]
    dup_source_ids = [sid for sid, cnt in Counter(source_ids).items() if cnt > 1]
    if dup_source_ids:
        raise ValueError(
            "Duplicate chapter source_id detected. "
            f"count={len(dup_source_ids)}, examples={dup_source_ids[:10]}"
        )

    return {"chapters_count": len(chapters)}


def validate_non_chapter_documents(documents: List[Document]) -> Dict[str, int]:
    if not documents:
        raise ValueError("No non-chapter documents prepared for vectorstore")

    _validate_metadata_shape(documents)
    _validate_duplicates(documents)

    by_source = Counter(d.metadata.get("source") for d in documents)
    return dict(by_source)
