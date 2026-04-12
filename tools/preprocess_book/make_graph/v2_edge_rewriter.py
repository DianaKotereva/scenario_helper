from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.utils.graph_search import AllBooksEdges, BookEdges, Description
from tools.preprocess_book.make_graph.v2_candidate_selector import normalize_text
from tools.preprocess_book.make_graph.v2_cluster_manager import ClusterManager


class EdgeRewriter:
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager

    def _resolve_cluster_id(
        self,
        value: str,
        source_hint: Optional[Set[int]] = None,
    ) -> str | None:
        candidates = self.cluster_manager.candidate_cluster_ids_for_name(value)
        if not candidates:
            return None
        hint = {int(v) for v in (source_hint or set()) if isinstance(v, int)}

        def _score(cluster_id: str) -> Tuple[int, int, int]:
            cluster = self.cluster_manager.clusters.get(cluster_id)
            if not cluster or not cluster.is_active:
                return (-1, -1, -1)
            cluster_sources = self.cluster_manager.cluster_source_ids(cluster_id)
            overlap = len(cluster_sources.intersection(hint)) if hint else 0
            exact_main = int(normalize_text(cluster.canonical_name) == normalize_text(value))
            support = len(cluster.event_ids)
            return (overlap, exact_main, support)

        return max(candidates, key=_score)

    @staticmethod
    def _iter_relation_evidence(rel: dict, default_source: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...]]]:
        """Normalize relation evidence from schema: descriptions=[{source_id, chapter_id, description, ...}, ...]."""
        out: List[Tuple[str, Tuple[int, ...]]] = []
        descriptions = rel.get("descriptions")
        if isinstance(descriptions, list):
            for item in descriptions:
                if not isinstance(item, dict):
                    continue
                desc = str(item.get("description", "")).strip()
                if not desc:
                    continue
                sid = item.get("source_id")
                if isinstance(sid, int):
                    out.append((desc, (sid,)))
                elif isinstance(sid, tuple) and sid:
                    out.append((desc, sid))
                elif isinstance(sid, list) and sid:
                    out.append((desc, tuple(int(v) for v in sid if isinstance(v, int))))
                else:
                    out.append((desc, default_source))
            return out
        return out

    def rewrite_relations(
        self,
        relations_by_source: Dict[Tuple[int, ...], List[dict]],
        cluster_to_node_key: Optional[Dict[str, str]] = None,
    ) -> AllBooksEdges:
        rel_graph = AllBooksEdges(relationships={})
        for source_id, rel_list in relations_by_source.items():
            for rel in rel_list:
                evidence_rows = self._iter_relation_evidence(rel, source_id)
                source_hint: Set[int] = set()
                for _desc_text, sid in evidence_rows:
                    source_hint.update(int(v) for v in sid if isinstance(v, int))

                src_cluster = self._resolve_cluster_id(
                    rel.get("source_node_id", ""),
                    source_hint=source_hint,
                )
                dst_cluster = self._resolve_cluster_id(
                    rel.get("target_node_id", ""),
                    source_hint=source_hint,
                )
                if not src_cluster or not dst_cluster:
                    continue

                src_cluster_obj = self.cluster_manager.clusters.get(src_cluster)
                dst_cluster_obj = self.cluster_manager.clusters.get(dst_cluster)
                if not src_cluster_obj or not dst_cluster_obj:
                    continue

                src = (
                    cluster_to_node_key.get(src_cluster, src_cluster_obj.canonical_name)
                    if cluster_to_node_key
                    else src_cluster_obj.canonical_name
                )
                dst = (
                    cluster_to_node_key.get(dst_cluster, dst_cluster_obj.canonical_name)
                    if cluster_to_node_key
                    else dst_cluster_obj.canonical_name
                )

                description_type = rel.get("type", "") or ""
                if src == dst:
                    # Keep explicit evidence when two aliases collapsed into the same canonical node.
                    description_type = "IDENTITY_MERGED_CONTEXT"

                pair = tuple(sorted([src, dst]))
                existing = rel_graph.relationships.get(
                    pair,
                    BookEdges(object_1=pair[0], object_2=pair[1], description=[]),
                )

                for raw_desc, desc_source_id in evidence_rows:
                    description = Description(
                        description=raw_desc,
                        type=description_type,
                        source_id=desc_source_id,
                    )
                    if description not in existing.description:
                        existing.description.append(description)
                rel_graph.relationships[pair] = existing
        return rel_graph
