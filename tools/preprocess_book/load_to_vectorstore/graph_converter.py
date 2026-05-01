"""
Convert BookGraph into LangChain documents.

Graph facts are grouped into bigger semantic blocks (by entity / relation pair
inside the same source chapter). Downstream splitter then chunks those blocks
by token size, so we avoid tiny low-signal vector documents.
"""

import hashlib
import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.utils.graph_search import BookGraph

logger = logging.getLogger(__name__)


GRAPH_SCHEMA_VERSION = "graph_v1"


def _to_int_source_id(source_id) -> int:
    if isinstance(source_id, int):
        return source_id
    if isinstance(source_id, tuple) and source_id:
        if isinstance(source_id[0], int):
            return source_id[0]
    raise ValueError(f"Unsupported source_id type: {source_id!r}")


def _slug(text: str) -> str:
    val = re.sub(r"\s+", "_", text.strip())
    val = re.sub(r"[^0-9A-Za-z_\-]+", "", val)
    return val[:80] or "unknown"


def _node_doc_uid(entity_name: str, source_id: int, fact_index: int) -> str:
    name_hash = hashlib.sha1(entity_name.encode("utf-8")).hexdigest()[:16]
    return f"node::{name_hash}::{source_id}::{fact_index}"


def _relation_doc_uid(
    object_1: str, object_2: str, source_id: int, fact_index: int
) -> str:
    rel_key = f"{object_1}::{object_2}"
    rel_hash = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()[:16]
    return f"rel::{rel_hash}::{source_id}::{fact_index}"


def _grouped_node_doc_uid(entity_name: str, source_id: int) -> str:
    name_hash = hashlib.sha1(entity_name.encode("utf-8")).hexdigest()[:16]
    return f"node_group::{name_hash}::{source_id}"


def _grouped_relation_doc_uid(object_1: str, object_2: str, source_id: int) -> str:
    rel_key = f"{object_1}::{object_2}"
    rel_hash = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()[:16]
    return f"rel_group::{rel_hash}::{source_id}"


def book_graph_to_documents(book_graph: BookGraph) -> List[Document]:
    """
    Convert BookGraph to grouped graph documents with strict metadata.
    """
    documents: List[Document] = []

    node_fact_count = 0
    relation_fact_count = 0

    node_groups: Dict[Tuple[str, int], Dict[str, object]] = defaultdict(
        lambda: {"classification": "", "facts": []}
    )
    relation_groups: Dict[Tuple[str, str, int], Dict[str, object]] = defaultdict(
        lambda: {"relation_type": "", "facts": []}
    )

    for node_name, node in book_graph.nodes.nodes.items():
        actions = getattr(node, "actions", []) or []
        for fact_index, action in enumerate(actions):
            action_text = getattr(action, "action", None)
            if not action_text or not str(action_text).strip():
                continue

            try:
                source_id = _to_int_source_id(getattr(action, "source_id", None))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skip invalid node fact source_id for entity '%s': %s",
                    node_name,
                    exc,
                )
                continue

            grouped = node_groups[(node_name, source_id)]
            grouped["classification"] = getattr(node, "classification", "")
            grouped["facts"].append(
                {
                    "fact_index": fact_index,
                    "text": str(action_text).strip(),
                    "doc_uid": _node_doc_uid(node_name, source_id, fact_index),
                }
            )
            node_fact_count += 1

    for relation in book_graph.relationships.relationships.values():
        descriptions = getattr(relation, "description", []) or []
        object_1 = getattr(relation, "object_1", "")
        object_2 = getattr(relation, "object_2", "")

        for fact_index, description in enumerate(descriptions):
            text = getattr(description, "description", None)
            if not text or not str(text).strip():
                continue

            try:
                source_id = _to_int_source_id(getattr(description, "source_id", None))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skip invalid relation fact source_id for '%s'->'%s': %s",
                    object_1,
                    object_2,
                    exc,
                )
                continue

            relation_type = getattr(description, "type", "") or ""
            grouped = relation_groups[(object_1, object_2, source_id)]
            if relation_type and not grouped["relation_type"]:
                grouped["relation_type"] = relation_type
            grouped["facts"].append(
                {
                    "fact_index": fact_index,
                    "text": str(text).strip(),
                    "doc_uid": _relation_doc_uid(object_1, object_2, source_id, fact_index),
                    "relation_type": relation_type,
                }
            )
            relation_fact_count += 1

    for (node_name, source_id), payload in node_groups.items():
        facts = sorted(payload["facts"], key=lambda x: int(x.get("fact_index", 0)))
        body = "\n".join(f"- {item['text']}" for item in facts if item.get("text"))
        if not body.strip():
            continue

        metadata = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "source": "nodes",
            "source_id": source_id,
            "chapter_id": source_id,
            "graph_doc_type": "node_fact_group",
            "entity_name": node_name,
            "entity_classification": payload.get("classification", ""),
            "fact_count": len(facts),
            "fact_doc_uids": [item.get("doc_uid", "") for item in facts],
            "doc_uid": _grouped_node_doc_uid(node_name, source_id),
            "name": node_name,
        }
        page_content = f"{node_name}\n{body}"
        documents.append(Document(page_content=page_content, metadata=metadata))

    for (object_1, object_2, source_id), payload in relation_groups.items():
        facts = sorted(payload["facts"], key=lambda x: int(x.get("fact_index", 0)))
        body = "\n".join(f"- {item['text']}" for item in facts if item.get("text"))
        if not body.strip():
            continue

        relation_type = str(payload.get("relation_type", "") or "")
        metadata = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "source": "relations",
            "source_id": source_id,
            "chapter_id": source_id,
            "graph_doc_type": "relation_fact_group",
            "relation_object_1": object_1,
            "relation_object_2": object_2,
            "relation_type": relation_type,
            "fact_count": len(facts),
            "fact_doc_uids": [item.get("doc_uid", "") for item in facts],
            "doc_uid": _grouped_relation_doc_uid(object_1, object_2, source_id),
            "name": (object_1, object_2),
        }
        title = f"{object_1} -> {object_2}"
        page_content = f"{title}\n{body}"
        documents.append(Document(page_content=page_content, metadata=metadata))

    logger.info(
        "Converted graph to grouped docs: node_facts=%s, relation_facts=%s, grouped_docs=%s",
        node_fact_count,
        relation_fact_count,
        len(documents),
    )
    return documents
