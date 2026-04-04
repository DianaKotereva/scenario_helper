from __future__ import annotations

import json
from typing import Dict, List, Tuple

from src.utils.graph_search import AllBooksEdges, BookEdges, Description
from tools.preprocess_book._v2.candidate_selector import normalize_text
from tools.preprocess_book._v2.cluster_manager import ClusterManager


class EdgeRewriter:
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager

    def _resolve_name(self, value: str) -> str | None:
        cluster_id = self.cluster_manager.name_to_cluster.get(normalize_text(value))
        if not cluster_id:
            return None
        root_id = self.cluster_manager.resolve_cluster_id(cluster_id)
        cluster = self.cluster_manager.clusters.get(root_id)
        if not cluster or not cluster.is_active:
            return None
        return cluster.canonical_name

    @staticmethod
    def _build_provenance(rel: dict, src_canonical: str, dst_canonical: str) -> str:
        source_raw = rel.get("source_node_id", "") or ""
        target_raw = rel.get("target_node_id", "") or ""
        payload = {
            "source_raw": source_raw,
            "target_raw": target_raw,
            "source_canonical": src_canonical,
            "target_canonical": dst_canonical,
            "source_aspect": source_raw if source_raw and source_raw != src_canonical else "",
            "target_aspect": target_raw if target_raw and target_raw != dst_canonical else "",
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _compose_description(rel: dict, provenance_json: str) -> str:
        raw_description = rel.get("description", "") or ""
        return f"{raw_description}\n[V2_PROVENANCE {provenance_json}]"

    def rewrite_relations(
        self, relations_by_source: Dict[Tuple[int, ...], List[dict]]
    ) -> AllBooksEdges:
        rel_graph = AllBooksEdges(relationships={})
        for source_id, rel_list in relations_by_source.items():
            for rel in rel_list:
                src = self._resolve_name(rel.get("source_node_id", ""))
                dst = self._resolve_name(rel.get("target_node_id", ""))
                if not src or not dst:
                    continue

                provenance_json = self._build_provenance(rel, src, dst)
                description_type = rel.get("type", "") or ""
                if src == dst:
                    # Keep explicit evidence when two aliases collapsed into the same canonical node.
                    description_type = "IDENTITY_MERGED_CONTEXT"

                pair = tuple(sorted([src, dst]))
                description = Description(
                    description=self._compose_description(rel, provenance_json),
                    type=description_type,
                    source_id=source_id,
                )
                existing = rel_graph.relationships.get(
                    pair,
                    BookEdges(object_1=pair[0], object_2=pair[1], description=[]),
                )
                if description not in existing.description:
                    existing.description.append(description)
                rel_graph.relationships[pair] = existing
        return rel_graph
