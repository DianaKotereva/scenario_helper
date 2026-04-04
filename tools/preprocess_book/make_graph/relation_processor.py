import logging
from typing import Any, Dict, List, Tuple

from src.utils.graph_search import AllBookNodes, AllBooksEdges, BookEdges, Description

logger = logging.getLogger(__name__)


class RelationProcessor:
    """Processor for graph relation records."""

    @staticmethod
    def _to_source_tuple(value: Any, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(value, int):
            return (value,)
        if isinstance(value, tuple) and value and isinstance(value[0], int):
            return value
        if isinstance(value, list) and value and isinstance(value[0], int):
            return tuple(value)
        return fallback

    @staticmethod
    def _format_relation_text(description: str, quotes: List[str], chapter_id: Any) -> str:
        desc = (description or "").strip()
        if not desc:
            return ""
        return desc

    @classmethod
    def _extract_descriptions(
        cls,
        rel: Dict[str, Any],
        source_id: Tuple[int, ...],
    ) -> List[Description]:
        fallback_sid = source_id[0] if source_id else 0
        rel_type = str(rel.get("type", "")).strip()

        raw_descriptions = rel.get("descriptions")
        if raw_descriptions is None and rel.get("description") is not None:
            raw_descriptions = rel.get("description")

        descriptions: List[Description] = []

        if isinstance(raw_descriptions, str):
            text = raw_descriptions.strip()
            if text:
                descriptions.append(
                    Description(description=text, type=rel_type, source_id=source_id)
                )
            return descriptions

        if not isinstance(raw_descriptions, list):
            return descriptions

        for item in raw_descriptions:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    descriptions.append(
                        Description(description=text, type=rel_type, source_id=source_id)
                    )
                continue

            if not isinstance(item, dict):
                continue

            sid = item.get("source_id", fallback_sid)
            cid = item.get("chapter_id", sid)
            desc = item.get("description")
            quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []

            text = cls._format_relation_text(str(desc or ""), quotes, cid)
            if not text:
                continue

            descriptions.append(
                Description(
                    description=text,
                    type=rel_type,
                    source_id=cls._to_source_tuple(sid, source_id),
                    chapter_id=int(cid) if isinstance(cid, int) else source_id[0],
                    quotes=[q.strip() for q in quotes if isinstance(q, str) and q.strip()],
                )
            )

        return descriptions

    @classmethod
    def process_relations(
        cls,
        all_book_nodes: AllBookNodes,
        rel_inputs: List[Dict],
        rel_graphs: AllBooksEdges,
        source_id: Tuple[int, ...],
    ) -> AllBooksEdges:
        all_mains = list(all_book_nodes.nodes.keys())
        rel_inputs = [item for item in rel_inputs if item]
        rel_inputs = [
            item
            for item in rel_inputs
            if item.get("source_node_id") in all_mains and item.get("target_node_id") in all_mains
        ]

        for rel in rel_inputs:
            try:
                source_node_id = rel.get("source_node_id")
                target_node_id = rel.get("target_node_id")

                if not source_node_id or not target_node_id:
                    logger.warning("Relation skipped due empty node ids: %s", rel)
                    continue

                pair = tuple(sorted([source_node_id, target_node_id]))
                if pair[0] == source_node_id:
                    object_1, object_2 = source_node_id, target_node_id
                else:
                    object_1, object_2 = target_node_id, source_node_id

                target_rel = rel_graphs.relationships.get(
                    pair,
                    BookEdges(object_1=object_1, object_2=object_2, description=[]),
                )

                for description in cls._extract_descriptions(rel, source_id):
                    if description not in target_rel.description:
                        target_rel.description.append(description)

                rel_graphs.relationships[pair] = target_rel
            except Exception as ex:
                logger.error(
                    "Error while processing relation for source_id=%s: %s",
                    source_id,
                    ex,
                    exc_info=True,
                )
                continue

        return rel_graphs
