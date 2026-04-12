import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.graph_search import AllBookNodes, AllBooksEdges
from tools.preprocess_book.storage.storage import FileManager

logger = logging.getLogger(__name__)


class BatchGraphBuilder:
    """
    Minimal batch-only graph builder for verification/merge stage.
    Uses already extracted batch payloads and applies existing node/relation processors.
    """

    def __init__(self, node_processor, relation_processor):
        self.node_processor = node_processor
        self.relation_processor = relation_processor

    @staticmethod
    def _sorted_batch_files(batch_payload_dir: Path, max_batches: Optional[int] = None) -> List[Path]:
        files = sorted(batch_payload_dir.glob("batch_*.json"))
        if max_batches is not None and max_batches > 0:
            files = files[:max_batches]
        return files

    @staticmethod
    def _parse_source_tuple_from_filename(path: Path) -> Tuple[int, ...]:
        # Expected format: batch_001_5_9.json
        m = re.match(r"batch_\d+_(\d+)_(\d+)\.json$", path.name)
        if not m:
            return tuple()
        start = int(m.group(1))
        end = int(m.group(2))
        if end < start:
            start, end = end, start
        return tuple(range(start, end + 1))

    @staticmethod
    def _extract_source_ids_from_payload(payload: Dict[str, Any]) -> Tuple[int, ...]:
        source_ids = set()

        for node in payload.get("nodes", []) if isinstance(payload.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            for action in node.get("actions", []) if isinstance(node.get("actions"), list) else []:
                if isinstance(action, dict) and isinstance(action.get("source_id"), int):
                    source_ids.add(int(action["source_id"]))

        for rel in payload.get("relations", []) if isinstance(payload.get("relations"), list) else []:
            if not isinstance(rel, dict):
                continue
            for desc in rel.get("descriptions", []) if isinstance(rel.get("descriptions"), list) else []:
                if isinstance(desc, dict) and isinstance(desc.get("source_id"), int):
                    source_ids.add(int(desc["source_id"]))

        return tuple(sorted(source_ids))

    @classmethod
    def _resolve_source_tuple(cls, payload: Dict[str, Any], path: Path) -> Tuple[int, ...]:
        # Prefer batch range from filename (authoritative expected chapters in batch).
        from_filename = cls._parse_source_tuple_from_filename(path)
        if from_filename:
            return from_filename
        # Fallback for non-standard filenames.
        return cls._extract_source_ids_from_payload(payload)

    @staticmethod
    def _extract_source_ids_from_graph(
        all_book_nodes: AllBookNodes,
        relation_graphs: AllBooksEdges,
    ) -> Tuple[int, ...]:
        source_ids = set()
        for node in all_book_nodes.nodes.values():
            for action in node.actions or []:
                sid = getattr(action, "source_id", None)
                if isinstance(sid, int):
                    source_ids.add(int(sid))
                elif isinstance(sid, tuple):
                    source_ids.update(int(v) for v in sid if isinstance(v, int))
                elif isinstance(sid, list):
                    source_ids.update(int(v) for v in sid if isinstance(v, int))

        for edge in relation_graphs.relationships.values():
            for desc in edge.description or []:
                sid = getattr(desc, "source_id", None)
                if isinstance(sid, int):
                    source_ids.add(int(sid))
                elif isinstance(sid, tuple):
                    source_ids.update(int(v) for v in sid if isinstance(v, int))
                elif isinstance(sid, list):
                    source_ids.update(int(v) for v in sid if isinstance(v, int))
        return tuple(sorted(source_ids))

    @staticmethod
    def _state_counts(all_book_nodes: AllBookNodes, relation_graphs: AllBooksEdges) -> Dict[str, int]:
        relation_fact_count = sum(len(edge.description or []) for edge in relation_graphs.relationships.values())
        node_fact_count = sum(len(node.actions or []) for node in all_book_nodes.nodes.values())
        return {
            "nodes_count": len(all_book_nodes.nodes),
            "relations_count": len(relation_graphs.relationships),
            "node_facts": node_fact_count,
            "relation_facts": relation_fact_count,
        }

    def build_graph_from_batch_payloads(
        self,
        batch_payload_dir: Path,
        max_batches: Optional[int] = None,
        snapshot_dir: Optional[Path] = None,
        strict_source_coverage: bool = True,
        strict_extraction_coverage: bool = True,
    ) -> Tuple[AllBookNodes, AllBooksEdges, List[Dict[str, Any]]]:
        all_book_nodes = AllBookNodes(nodes={}, names_list={})
        relation_graphs = AllBooksEdges(relationships={})
        cumulative_expected_source_ids = set()

        batch_files = self._sorted_batch_files(batch_payload_dir, max_batches=max_batches)
        if not batch_files:
            logger.warning("No batch payload files found in %s", batch_payload_dir)
            return all_book_nodes, relation_graphs, []

        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)

        reports: List[Dict[str, Any]] = []
        for idx, batch_file in enumerate(batch_files):
            payload = json.loads(batch_file.read_text(encoding="utf-8"))
            nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
            relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []
            source_tuple = self._resolve_source_tuple(payload, batch_file)

            if not source_tuple:
                logger.warning("Skip batch without source tuple: %s", batch_file)
                continue

            # Extraction coverage gate:
            # each source_id from the batch range must be present in extracted evidences.
            extracted_source_ids = set(self._extract_source_ids_from_payload(payload))
            expected_source_ids = set(source_tuple)
            missing_in_extraction = sorted(expected_source_ids - extracted_source_ids)
            if missing_in_extraction:
                message = (
                    "Extraction coverage gate failed at batch idx="
                    f"{idx}: missing source_ids in payload={missing_in_extraction}. "
                    f"batch_file={batch_file}. expected={sorted(expected_source_ids)} "
                    f"extracted={sorted(extracted_source_ids)}"
                )
                if strict_extraction_coverage:
                    raise RuntimeError(message)
                logger.warning(message)

            before = self._state_counts(all_book_nodes, relation_graphs)
            all_book_nodes = self.node_processor.process_input(
                input_data=nodes,
                rel_inputs=relations,
                source_id=source_tuple,
                all_book_nodes=all_book_nodes,
            )
            relation_graphs = self.relation_processor.process_relations(
                all_book_nodes=all_book_nodes,
                rel_inputs=relations,
                rel_graphs=relation_graphs,
                source_id=source_tuple,
            )
            after = self._state_counts(all_book_nodes, relation_graphs)
            expected_current = set(source_tuple)
            cumulative_expected_source_ids.update(expected_current)
            graph_source_ids = set(self._extract_source_ids_from_graph(all_book_nodes, relation_graphs))
            missing_current = sorted(expected_current - graph_source_ids)
            missing_cumulative = sorted(cumulative_expected_source_ids - graph_source_ids)

            report = {
                "batch_index": idx,
                "batch_file": str(batch_file),
                "source_ids": list(source_tuple),
                "input_nodes": len(nodes),
                "input_relations": len(relations),
                "before": before,
                "after": after,
                "delta_nodes": after["nodes_count"] - before["nodes_count"],
                "delta_relations": after["relations_count"] - before["relations_count"],
                "delta_node_facts": after["node_facts"] - before["node_facts"],
                "delta_relation_facts": after["relation_facts"] - before["relation_facts"],
                "extracted_source_ids_current": sorted(extracted_source_ids),
                "missing_source_ids_in_extraction": missing_in_extraction,
                "expected_source_ids_current": sorted(expected_current),
                "expected_source_ids_cumulative": sorted(cumulative_expected_source_ids),
                "graph_source_ids_present": sorted(graph_source_ids),
                "missing_source_ids_current": missing_current,
                "missing_source_ids_cumulative": missing_cumulative,
            }
            reports.append(report)

            if strict_source_coverage and missing_current:
                raise RuntimeError(
                    "Source coverage gate failed at batch idx="
                    f"{idx}: missing current source_ids={missing_current}. "
                    f"batch_file={batch_file}"
                )

            if snapshot_dir is not None:
                FileManager.save_pickle(
                    {
                        "batch_report": report,
                        "all_book_nodes": all_book_nodes,
                        "relation_graphs": relation_graphs,
                    },
                    snapshot_dir / f"graph_state_after_batch_{idx:03d}.pkl",
                )
                (snapshot_dir / f"graph_state_after_batch_{idx:03d}.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        return all_book_nodes, relation_graphs, reports
