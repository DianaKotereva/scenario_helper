"""
Convert BookGraph into atomic LangChain documents.

Each graph fact is represented as exactly one document:
- one node action = one document
- one relation description = one document
"""

import logging
import re
import hashlib
from typing import List

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


def book_graph_to_documents(book_graph: BookGraph) -> List[Document]:
    """
    Convert BookGraph to atomic graph documents with strict metadata.
    """
    documents: List[Document] = []

    node_fact_count = 0
    relation_fact_count = 0

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

            metadata = {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "source": "nodes",
                "source_id": source_id,
                "graph_doc_type": "node_fact",
                "entity_name": node_name,
                "entity_classification": getattr(node, "classification", ""),
                "fact_index": fact_index,
                "doc_uid": _node_doc_uid(node_name, source_id, fact_index),
                # Backward-compatible field used in retrieval/debug tooling.
                "name": node_name,
            }

            documents.append(Document(page_content=str(action_text).strip(), metadata=metadata))
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
            metadata = {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "source": "relations",
                "source_id": source_id,
                "graph_doc_type": "relation_fact",
                "relation_object_1": object_1,
                "relation_object_2": object_2,
                "relation_type": relation_type,
                "fact_index": fact_index,
                "doc_uid": _relation_doc_uid(object_1, object_2, source_id, fact_index),
                # Backward-compatible field used in retrieval/debug tooling.
                "name": (object_1, object_2),
            }
            documents.append(Document(page_content=str(text).strip(), metadata=metadata))
            relation_fact_count += 1

    logger.info(
        "Converted graph to atomic docs: node_facts=%s, relation_facts=%s, total=%s",
        node_fact_count,
        relation_fact_count,
        len(documents),
    )
    return documents
