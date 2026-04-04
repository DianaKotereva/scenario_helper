import ast
import asyncio
import logging
import os
from typing import List, Optional, Set, Tuple

from src.utils.graph_search import AllBookNodes, AllBooksEdges
from tools.preprocess_book.config.preprocess_settings import (
    GRAPH_BUILD_CONCURRENCY,
    GRAPH_NODES_DIR,
    GRAPH_RELATIONS_DIR,
    RESULTS_DIR,
)
from tools.preprocess_book.make_graph.node_processor import NodeProcessor
from tools.preprocess_book.make_graph.relation_processor import RelationProcessor
from tools.preprocess_book.storage.storage import FileManager

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, node_processor: NodeProcessor, relation_processor: RelationProcessor):
        self.node_processor = node_processor
        self.relation_processor = relation_processor

    def _extract_source_id(self, filename: str) -> Tuple[int, ...]:
        """Extract source_id tuple from result filename."""
        source_id_str = filename.replace(".pkl", "")
        try:
            source_id = ast.literal_eval(source_id_str)
            if isinstance(source_id, tuple):
                return source_id
            return (int(source_id),)
        except (ValueError, SyntaxError, TypeError):
            parts = source_id_str.split("_")
            source_id = tuple(int(p) for p in parts if p.isdigit())
            if not source_id:
                raise ValueError(f"Failed to parse source_id from {filename}")
            return source_id

    def _sort_key(self, filename: str) -> int:
        try:
            return int(self._extract_source_id(filename)[0])
        except Exception:
            return 0

    def _get_sorted_result_files(
        self,
        max_files: Optional[int] = None,
        allowed_source_ids: Optional[Set[Tuple[int, ...]]] = None,
    ) -> List[str]:
        if not RESULTS_DIR.exists():
            logger.error(f"Results directory not found: {RESULTS_DIR}")
            return []

        result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl")]
        if not result_files:
            logger.warning(f"No result files found in {RESULTS_DIR}")
            return []

        if allowed_source_ids is not None:
            filtered: List[str] = []
            for filename in result_files:
                try:
                    sid = self._extract_source_id(filename)
                except Exception:
                    continue
                if sid in allowed_source_ids:
                    filtered.append(filename)
            result_files = filtered

        result_files = sorted(result_files, key=self._sort_key)

        if max_files and max_files > 0:
            result_files = result_files[:max_files]

        return result_files

    async def build_graph_from_results_async(
        self,
        concurrency: int = 10,
        max_files: Optional[int] = None,
        allowed_source_ids: Optional[Set[Tuple[int, ...]]] = None,
    ) -> Tuple[AllBookNodes, AllBooksEdges]:
        """
        Build graph from extraction results.

        File reads/writes are moved to threads so concurrency actually helps IO-bound work.
        Graph merge stays under lock for consistency.
        """
        all_book_nodes = AllBookNodes()
        relation_graphs = AllBooksEdges(relationships={})
        graph_lock = asyncio.Lock()

        result_files = self._get_sorted_result_files(
            max_files=max_files,
            allowed_source_ids=allowed_source_ids,
        )
        if not result_files:
            return all_book_nodes, relation_graphs

        async def process_file(filename: str) -> bool:
            try:
                source_id = self._extract_source_id(filename)
                path = RESULTS_DIR / filename
                if not path.exists():
                    logger.warning(f"Result file not found: {path}")
                    return False

                res = await asyncio.to_thread(FileManager.load_pickle, path)
                if not isinstance(res, dict):
                    logger.warning(f"Invalid result format for {filename}: expected dict")
                    return False

                nodes = res.get("nodes", [])
                relations = res.get("relations", [])

                async with graph_lock:
                    nonlocal all_book_nodes, relation_graphs
                    all_book_nodes = self.node_processor.process_input(
                        input_data=nodes,
                        rel_inputs=relations,
                        source_id=source_id,
                        all_book_nodes=all_book_nodes,
                    )
                    relation_graphs = self.relation_processor.process_relations(
                        all_book_nodes=all_book_nodes,
                        rel_inputs=relations,
                        rel_graphs=relation_graphs,
                        source_id=source_id,
                    )

                    await asyncio.to_thread(
                        FileManager.save_pickle,
                        all_book_nodes,
                        GRAPH_NODES_DIR / filename,
                    )
                    await asyncio.to_thread(
                        FileManager.save_pickle,
                        relation_graphs,
                        GRAPH_RELATIONS_DIR / filename,
                    )
                    if source_id and isinstance(source_id[0], int) and source_id[0] % 5 == 0:
                        logger.info(
                            "Graph build progress marker: chapter source_id=%s",
                            source_id[0],
                        )
                return True
            except Exception as ex:  # noqa: BLE001
                logger.error(f"Error processing {filename}: {ex}", exc_info=True)
                return False

        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_limit(filename: str) -> bool:
            async with semaphore:
                return await process_file(filename)

        logger.info(
            "Start graph build from %s files with concurrency=%s",
            len(result_files),
            concurrency,
        )
        results = await asyncio.gather(*(process_with_limit(f) for f in result_files))
        logger.info("Successfully processed %s/%s files", sum(results), len(result_files))
        return all_book_nodes, relation_graphs

    def build_graph_from_results(
        self,
        concurrency: Optional[int] = None,
        max_files: Optional[int] = None,
        allowed_source_ids: Optional[Set[Tuple[int, ...]]] = None,
    ) -> Tuple[AllBookNodes, AllBooksEdges]:
        concurrency = concurrency or GRAPH_BUILD_CONCURRENCY
        return asyncio.run(
            self.build_graph_from_results_async(
                concurrency=concurrency,
                max_files=max_files,
                allowed_source_ids=allowed_source_ids,
            )
        )
