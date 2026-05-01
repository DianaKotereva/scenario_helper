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
    ) -> tuple[AllBooksEdges, Dict[str, object]]:
        rel_graph = AllBooksEdges(relationships={})
        dropped_rows: List[Dict[str, object]] = []
        per_source_metrics: Dict[str, Dict[str, int]] = {}

        def _source_key(source_id: Tuple[int, ...]) -> str:
            return ",".join(str(v) for v in source_id)

        def _metric_row(source_id: Tuple[int, ...]) -> Dict[str, int]:
            key = _source_key(source_id)
            if key not in per_source_metrics:
                per_source_metrics[key] = {
                    "extracted_relations": 0,
                    "rewritten_relations": 0,
                    "dropped_relations": 0,
                    "self_merged_relations": 0,
                }
            return per_source_metrics[key]

        def _resolve_with_meta(value: str, source_hint: Set[int]) -> tuple[str | None, str | None]:
            candidates = self.cluster_manager.candidate_cluster_ids_for_name(value)
            if not candidates:
                return None, "missing_endpoint"

            active: List[str] = []
            for cluster_id in candidates:
                cluster = self.cluster_manager.clusters.get(cluster_id)
                if cluster and cluster.is_active:
                    active.append(cluster_id)
            if not active:
                return None, "missing_endpoint"

            if len(active) > 1:
                scored = []
                for cluster_id in active:
                    cluster = self.cluster_manager.clusters.get(cluster_id)
                    cluster_sources = self.cluster_manager.cluster_source_ids(cluster_id)
                    overlap = len(cluster_sources.intersection(source_hint)) if source_hint else 0
                    exact_main = int(
                        normalize_text(cluster.canonical_name) == normalize_text(value)
                    ) if cluster else 0
                    support = len(cluster.event_ids) if cluster else 0
                    scored.append((cluster_id, overlap, exact_main, support))
                scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
                best = scored[0]
                second = scored[1]
                # Ambiguous when top-2 have equal ranking signals.
                if (best[1], best[2], best[3]) == (second[1], second[2], second[3]):
                    return None, "ambiguous_endpoint"

            resolved = self._resolve_cluster_id(value, source_hint=source_hint)
            if not resolved:
                return None, "missing_endpoint"
            return resolved, None

        for source_id, rel_list in relations_by_source.items():
            metric = _metric_row(source_id)
            for rel in rel_list:
                metric["extracted_relations"] += 1
                evidence_rows = self._iter_relation_evidence(rel, source_id)
                source_hint: Set[int] = set()
                for _desc_text, sid in evidence_rows:
                    source_hint.update(int(v) for v in sid if isinstance(v, int))

                src_name = rel.get("source_node_id", "") or ""
                dst_name = rel.get("target_node_id", "") or ""
                src_cluster, src_err = _resolve_with_meta(src_name, source_hint)
                dst_cluster, dst_err = _resolve_with_meta(dst_name, source_hint)
                if not src_cluster or not dst_cluster:
                    reason = "missing_source_endpoint"
                    if src_err == "ambiguous_endpoint":
                        reason = "ambiguous_endpoint"
                    elif dst_err == "ambiguous_endpoint":
                        reason = "ambiguous_endpoint"
                    elif src_err is None and dst_err is not None:
                        reason = "missing_target_endpoint"
                    elif src_err is not None and dst_err is None:
                        reason = "missing_source_endpoint"
                    elif src_err is not None and dst_err is not None:
                        reason = "missing_source_endpoint"
                    dropped_rows.append(
                        {
                            "source_id": list(source_id),
                            "reason": reason,
                            "source_node_id": str(src_name),
                            "target_node_id": str(dst_name),
                            "type": str(rel.get("type", "") or ""),
                        }
                    )
                    metric["dropped_relations"] += 1
                    continue

                src_cluster_obj = self.cluster_manager.clusters.get(src_cluster)
                dst_cluster_obj = self.cluster_manager.clusters.get(dst_cluster)
                if not src_cluster_obj or not dst_cluster_obj:
                    dropped_rows.append(
                        {
                            "source_id": list(source_id),
                            "reason": "low_confidence",
                            "source_node_id": str(src_name),
                            "target_node_id": str(dst_name),
                            "type": str(rel.get("type", "") or ""),
                        }
                    )
                    metric["dropped_relations"] += 1
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
                    metric["self_merged_relations"] += 1

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
                metric["rewritten_relations"] += 1
        report: Dict[str, object] = {
            "dropped_relations": dropped_rows,
            "per_source_metrics": per_source_metrics,
        }
        return rel_graph, report
