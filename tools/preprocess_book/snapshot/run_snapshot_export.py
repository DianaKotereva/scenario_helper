"""CLI for snapshot export + validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .exporter import export_snapshot
from .validator import validate_snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export snapshot jsonl files from preprocessing artifacts")
    parser.add_argument("--book-path", type=str, required=True, help="Path to source book txt")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for snapshot jsonl files")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory with raw extraction results/*.pkl")
    parser.add_argument("--graph-nodes-dir", type=str, default=None, help="Directory with graph_nodes/*.pkl")
    parser.add_argument("--graph-relations-dir", type=str, default=None, help="Directory with graph_relations/*.pkl")
    parser.add_argument("--merged-graph-path", type=str, default=None, help="Path to merged bookgraph.pkl")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of chapter files (0 = all)")
    parser.add_argument("--chunk-size", type=int, default=1400, help="Chunk size for chunks.jsonl")
    parser.add_argument("--chunk-overlap", type=int, default=250, help="Chunk overlap for chunks.jsonl")
    parser.add_argument("--validate", action="store_true", help="Run validation and save validation_report.json")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Exit with non-zero code if validation status is fail",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export_stats = export_snapshot(
        output_dir=output_dir,
        book_path=Path(args.book_path),
        max_files=args.max_files,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        graph_nodes_dir=Path(args.graph_nodes_dir) if args.graph_nodes_dir else None,
        graph_relations_dir=Path(args.graph_relations_dir) if args.graph_relations_dir else None,
        merged_graph_path=Path(args.merged_graph_path) if args.merged_graph_path else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    report = {"export": export_stats}
    if args.validate:
        validation_report = validate_snapshot(output_dir)
        report["validation"] = validation_report
        report_path = output_dir / "validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Validation report saved: {report_path}")
        if args.strict_validation and validation_report.get("status") != "ok":
            print("Validation failed.")
            sys.exit(1)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
