#!/usr/bin/env python3
"""
Production indexing path from final batch-merged graph.

Indexes:
- graph/book/summaries chunks -> vector index
- full chapters -> chapters OpenSearch index
- deterministic snapshot jsonl (+ optional entity profile summaries)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from tools.preprocess_book.config.preprocess_settings import (
    GRAPH_NODES_DIR,
    GRAPH_RELATIONS_DIR,
    SUMMARIES_DIR,
    VECTORSTORE_CHUNK_SIZE,
    VECTORSTORE_PICKLE_PATH,
)
from tools.preprocess_book.load_to_vectorstore import (
    ChapterIndexer,
    DocumentPreparer,
    VectorStoreLoader,
)
from tools.preprocess_book.snapshot import export_snapshot, validate_snapshot
from tools.preprocess_book.snapshot.profile_summarizer import generate_entity_profiles
from tools.preprocess_book.storage.storage import FileManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _graph_source_ids(book_graph) -> set[int]:
    out: set[int] = set()
    for node in book_graph.nodes.nodes.values():
        for action in (getattr(node, "actions", None) or []):
            sid = getattr(action, "source_id", None)
            if isinstance(sid, tuple):
                out.update(int(v) for v in sid if isinstance(v, int))
            elif isinstance(sid, int):
                out.add(int(sid))
    for edge in book_graph.relationships.relationships.values():
        for desc in (getattr(edge, "description", None) or []):
            sid = getattr(desc, "source_id", None)
            if isinstance(sid, tuple):
                out.update(int(v) for v in sid if isinstance(v, int))
            elif isinstance(sid, int):
                out.add(int(sid))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index OpenSearch from final batch graph")
    parser.add_argument("--bookgraph-path", type=str, required=True, help="Path to final bookgraph_from_batches.pkl")
    parser.add_argument("--book-path", type=str, required=True, help="Path to source book txt")
    parser.add_argument(
        "--summaries-dir",
        type=str,
        default=str(SUMMARIES_DIR),
        help="Summaries directory (optional)",
    )
    parser.add_argument(
        "--vectorstore-pickle-path",
        type=str,
        default=str(VECTORSTORE_PICKLE_PATH),
        help="Pickle path for vectorstore docs",
    )
    parser.add_argument("--force-reload", action="store_true", help="Force full index reload")
    parser.add_argument(
        "--skip-vector-index",
        action="store_true",
        help="Skip vector indexing step (chapters + snapshot still run)",
    )
    parser.add_argument(
        "--snapshot-output-dir",
        type=str,
        required=True,
        help="Snapshot output directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional results directory for raw snapshot rows",
    )
    parser.add_argument(
        "--graph-nodes-dir",
        type=str,
        default=str(GRAPH_NODES_DIR),
        help="Fallback graph_nodes/*.pkl directory for snapshot export",
    )
    parser.add_argument(
        "--graph-relations-dir",
        type=str,
        default=str(GRAPH_RELATIONS_DIR),
        help="Fallback graph_relations/*.pkl directory for snapshot export",
    )
    parser.add_argument("--validate-snapshot", action="store_true")
    parser.add_argument("--generate-profiles", action="store_true")
    parser.add_argument("--profile-llm-type", type=str, default="deepseek")
    parser.add_argument("--profile-concurrency", type=int, default=20)
    parser.add_argument(
        "--strict-vector-index",
        action="store_true",
        help="Fail the whole run if vector indexing fails",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bookgraph_path = Path(args.bookgraph_path)
    book_path = Path(args.book_path)
    summaries_dir = Path(args.summaries_dir)
    vectorstore_pickle_path = Path(args.vectorstore_pickle_path)
    snapshot_output_dir = Path(args.snapshot_output_dir)

    logger.info("Loading merged graph: %s", bookgraph_path)
    book_graph = FileManager.load_pickle(bookgraph_path)
    graph_source_ids = _graph_source_ids(book_graph)
    logger.info("Graph source ids: %s", len(graph_source_ids))

    logger.info("Preparing documents (book + summaries + graph)")
    preparer = DocumentPreparer(chunk_size=VECTORSTORE_CHUNK_SIZE)
    all_documents = preparer.prepare_all_documents(
        book_path=book_path,
        summaries_dir=summaries_dir if summaries_dir.exists() else None,
        book_graph=book_graph,
        include_book=True,
        include_summaries=summaries_dir.exists(),
        include_graph=True,
    )

    chapters = preparer.get_chapters(all_documents)
    if graph_source_ids:
        chapters = [
            doc
            for doc in chapters
            if isinstance(doc.metadata.get("source_id"), int)
            and int(doc.metadata["source_id"]) in graph_source_ids
        ]
    other_documents = [doc for doc in all_documents if doc not in chapters]
    if graph_source_ids:
        other_documents = [
            doc
            for doc in other_documents
            if not isinstance(doc.metadata.get("source_id"), int)
            or int(doc.metadata["source_id"]) in graph_source_ids
        ]
    split_docs = preparer.split_to_small_chunks(other_documents)

    vector_index_status = {
        "ok": True,
        "skipped": args.skip_vector_index,
        "docs_indexed": 0 if args.skip_vector_index else len(split_docs),
        "error": None,
    }
    if args.skip_vector_index:
        logger.warning("Skipping vector indexing by flag --skip-vector-index")
    else:
        logger.info("Indexing vector documents: %s", len(split_docs))
        try:
            loader = VectorStoreLoader()
            loader.load_documents(
                documents=split_docs,
                output_pickle_path=vectorstore_pickle_path,
                force_reload=args.force_reload,
            )
        except Exception as exc:  # noqa: BLE001
            vector_index_status["ok"] = False
            vector_index_status["error"] = str(exc)
            logger.exception("Vector indexing failed: %s", exc)
            if args.strict_vector_index:
                raise

    logger.info("Indexing chapters: %s", len(chapters))
    chapter_indexer = ChapterIndexer()
    chapter_indexer.index_chapters(chapters=chapters, force_reload=args.force_reload)

    logger.info("Exporting deterministic snapshot: %s", snapshot_output_dir)
    export_kwargs = {
        "output_dir": snapshot_output_dir,
        "book_path": book_path,
        "max_files": 0,
        "results_dir": Path(args.results_dir) if args.results_dir else None,
        "graph_nodes_dir": Path(args.graph_nodes_dir) if args.graph_nodes_dir else None,
        "graph_relations_dir": Path(args.graph_relations_dir) if args.graph_relations_dir else None,
        "merged_graph_path": bookgraph_path,
    }
    export_stats = export_snapshot(**export_kwargs)

    profile_stats = None
    if args.generate_profiles:
        profile_stats = asyncio.run(
            generate_entity_profiles(
                snapshot_dir=snapshot_output_dir,
                llm_type=args.profile_llm_type,
                concurrency=args.profile_concurrency,
            )
        )

    validation = None
    if args.validate_snapshot:
        validation = validate_snapshot(snapshot_output_dir)
        (snapshot_output_dir / "validation_report.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "bookgraph_path": str(bookgraph_path),
        "vectorstore_pickle_path": str(vectorstore_pickle_path),
        "snapshot_output_dir": str(snapshot_output_dir),
        "vector_index_status": vector_index_status,
        "export_stats": export_stats,
        "profile_stats": profile_stats,
        "validation": validation,
    }
    (snapshot_output_dir / "indexing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Indexing finished. Report: %s", snapshot_output_dir / "indexing_report.json")


if __name__ == "__main__":
    main()
