import argparse
import json
import logging
import shutil
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser

from src.utils.graph_search import BookGraph
from tools.preprocess_book.make_graph.batch_graph_builder import BatchGraphBuilder
from tools.preprocess_book.make_graph.node_processor import NodeProcessor
from tools.preprocess_book.make_graph.relation_processor import RelationProcessor
from tools.preprocess_book.make_graph.verification_service import VerificationService
from tools.preprocess_book.prompts.verificator import Verification
from tools.preprocess_book.storage.storage import FileManager
from tools.preprocess_book.utils.llm_factory import create_llm

logger = logging.getLogger(__name__)


def _copy_extraction_batch_files(batch_files: list[Path], extraction_out_dir: Path) -> None:
    extraction_out_dir.mkdir(parents=True, exist_ok=True)
    for fp in batch_files:
        shutil.copy2(fp, extraction_out_dir / fp.name)


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "batch_verification.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-only verification/merge runner.")
    parser.add_argument(
        "--batch-payload-dir",
        type=str,
        required=True,
        help="Directory with extraction batch payloads (batch_*.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store artifacts: copied extractions, merge snapshots, final graph.",
    )
    parser.add_argument("--max-batches", type=int, default=3, help="How many batch payloads to process.")
    parser.add_argument("--llm-type", type=str, default="deepseek", help="LLM provider for verification.")
    parser.add_argument(
        "--no-strict-source-coverage",
        action="store_true",
        help="Disable strict per-batch source coverage gate after merge.",
    )
    parser.add_argument(
        "--no-strict-extraction-coverage",
        action="store_true",
        help="Disable strict extraction coverage gate (missing source_ids in payload).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _configure_logging(output_dir)

    batch_payload_dir = Path(args.batch_payload_dir)
    batch_files = sorted(batch_payload_dir.glob("batch_*.json"))
    if args.max_batches > 0:
        batch_files = batch_files[: args.max_batches]
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.json in {batch_payload_dir}")

    extraction_out_dir = output_dir / "extractions"
    merge_snapshots_dir = output_dir / "merge_snapshots"
    _copy_extraction_batch_files(batch_files, extraction_out_dir)

    llm = create_llm(llm_type=args.llm_type)
    verificator = Verification(llm=llm, parser=JsonOutputParser())
    verification_service = VerificationService(verificator)
    node_processor = NodeProcessor(verification_service)
    relation_processor = RelationProcessor()
    builder = BatchGraphBuilder(node_processor=node_processor, relation_processor=relation_processor)

    all_book_nodes, relation_graphs, reports = builder.build_graph_from_batch_payloads(
        batch_payload_dir=batch_payload_dir,
        max_batches=args.max_batches,
        snapshot_dir=merge_snapshots_dir,
        strict_source_coverage=not args.no_strict_source_coverage,
        strict_extraction_coverage=not args.no_strict_extraction_coverage,
    )

    final_graph = BookGraph(nodes=all_book_nodes, relationships=relation_graphs)
    final_graph_path = output_dir / "bookgraph_from_batches.pkl"
    FileManager.save_pickle(final_graph, final_graph_path)

    report_path = output_dir / "merge_report.json"
    report_payload = {
        "batch_payload_dir": str(batch_payload_dir),
        "processed_batches": len(reports),
        "reports": reports,
        "final_nodes": len(all_book_nodes.nodes),
        "final_relations": len(relation_graphs.relationships),
        "final_node_facts": sum(len(node.actions or []) for node in all_book_nodes.nodes.values()),
        "final_relation_facts": sum(
            len(edge.description or []) for edge in relation_graphs.relationships.values()
        ),
        "final_graph_path": str(final_graph_path),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Batch verification finished. Artifacts saved to %s", output_dir)
    logger.info("Processed batches: %s", len(reports))
    logger.info("Final graph: nodes=%s relations=%s", len(all_book_nodes.nodes), len(relation_graphs.relationships))


if __name__ == "__main__":
    main()
