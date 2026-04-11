from __future__ import annotations

import json
from typing import Dict, List, Tuple

from src.utils.graph_search import AllBooksEdges, BookEdges, Description
from tools.preprocess_book.make_graph.v2_candidate_selector import normalize_text
from tools.preprocess_book.make_graph.v2_cluster_manager import ClusterManager


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
                existing = rel_graph.relationships.get(
                    pair,
                    BookEdges(object_1=pair[0], object_2=pair[1], description=[]),
                )

                for raw_desc, desc_source_id in self._iter_relation_evidence(rel, source_id):
                    description = Description(
                        description=f"{raw_desc}\n[V2_PROVENANCE {provenance_json}]",
                        type=description_type,
                        source_id=desc_source_id,
                    )
                    if description not in existing.description:
                        existing.description.append(description)
                rel_graph.relationships[pair] = existing
        return rel_graph
