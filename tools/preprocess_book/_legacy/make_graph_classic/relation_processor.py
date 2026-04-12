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

    @staticmethod
    def _name_match(a: str, b: str) -> bool:
        a_l = (a or "").strip().lower()
        b_l = (b or "").strip().lower()
        if not a_l or not b_l:
            return False
        return a_l in b_l or b_l in a_l

    @staticmethod
    def _node_source_ids(node_obj) -> set[int]:
        source_ids: set[int] = set()
        for action in getattr(node_obj, "actions", []) or []:
            sid = getattr(action, "source_id", None)
            if isinstance(sid, int):
                source_ids.add(int(sid))
            elif isinstance(sid, tuple):
                source_ids.update(int(v) for v in sid if isinstance(v, int))
            elif isinstance(sid, list):
                source_ids.update(int(v) for v in sid if isinstance(v, int))
        return source_ids

    @classmethod
    def _canonicalize_node_id(
        cls,
        node_id: Any,
        all_book_nodes: AllBookNodes,
        source_id: Tuple[int, ...],
    ) -> str:
        raw_name = str(node_id or "").strip()
        if not raw_name:
            return ""

        # Primary canonicalization path via alias index.
        mapped = all_book_nodes.names_list.get(raw_name)
        mapped_key = mapped if isinstance(mapped, str) and mapped in all_book_nodes.nodes else ""

        # Fallback: fuzzy candidate search in nodes.
        candidates: List[str] = []
        for key, node_obj in all_book_nodes.nodes.items():
            names = [getattr(node_obj, "main_name", "")] + list(getattr(node_obj, "alt_names", []) or [])
            if any(cls._name_match(raw_name, n) for n in names):
                candidates.append(key)

        requested_sources = {int(v) for v in source_id if isinstance(v, int)}
        if mapped_key and (not requested_sources):
            return mapped_key

        if mapped_key:
            mapped_overlap = len(cls._node_source_ids(all_book_nodes.nodes[mapped_key]).intersection(requested_sources))
            # Keep alias mapping when source context does not contradict it.
            if mapped_overlap > 0:
                return mapped_key

        if not candidates:
            return mapped_key
        if len(candidates) == 1:
            return candidates[0]

        if mapped_key and mapped_key not in candidates:
            candidates.append(mapped_key)

        def _score(node_key: str) -> tuple[int, int, int]:
            node_obj = all_book_nodes.nodes[node_key]
            node_sources = cls._node_source_ids(node_obj)
            overlap = len(node_sources.intersection(requested_sources))
            exact_main = int(raw_name.lower() == str(getattr(node_obj, "main_name", "")).lower())
            exact_alt = int(raw_name.lower() in {str(v).lower() for v in (getattr(node_obj, "alt_names", []) or [])})
            return overlap, exact_main, exact_alt

        return max(candidates, key=_score)

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
        rel_inputs = [item for item in rel_inputs if item]

        for rel in rel_inputs:
            try:
                source_node_id = cls._canonicalize_node_id(
                    rel.get("source_node_id"),
                    all_book_nodes=all_book_nodes,
                    source_id=source_id,
                )
                target_node_id = cls._canonicalize_node_id(
                    rel.get("target_node_id"),
                    all_book_nodes=all_book_nodes,
                    source_id=source_id,
                )

                if not source_node_id or not target_node_id:
                    logger.warning(
                        "Relation skipped due unresolved node ids: source=%s target=%s raw=%s",
                        rel.get("source_node_id"),
                        rel.get("target_node_id"),
                        rel,
                    )
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
