#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tools.preprocess_book._v2.io import make_run_prefix
from tools.preprocess_book._v2.quality_gates import evaluate_merge_quality
from tools.preprocess_book._v2.verification_pipeline import VerificationPipelineV2
from tools.preprocess_book.storage.storage import FileManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run verification-only v2 pipeline on precomputed extraction results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="tools/preprocess_book/_v2/results",
        help="Directory with precomputed extraction *.pkl files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output BookGraph pickle",
    )
    parser.add_argument(
        "--logs-path",
        type=str,
        default=None,
        help="Path for JSONL decisions log",
    )
    parser.add_argument("--max-files", type=int, default=10, help="Limit number of chapters")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate top-k for LLM judge")
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=6,
        help="Async concurrency for LLM judge calls inside candidate verification",
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        default=None,
        help="LLM type override (deepseek/openai)",
    )
    parser.add_argument(
        "--disable-llm-judge",
        action="store_true",
        help="Disable LLM judge (debug only)",
    )
    parser.add_argument(
        "--disable-bridge-merge",
        action="store_true",
        help="Disable cluster bridge merge pass",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow overwriting output pickle if it exists",
    )
    parser.add_argument(
        "--baseline-graph",
        type=str,
        default=None,
        help="Optional baseline AS-IS graph pickle for merge-quality comparison",
    )
    parser.add_argument(
        "--enforce-merge-gates",
        action="store_true",
        help="Fail run when merge-quality gates detect degradation",
    )
    parser.add_argument(
        "--quality-report-path",
        type=str,
        default=None,
        help="Path to save merge-quality gate report (JSON)",
    )
    parser.add_argument(
        "--max-relation-drop-ratio",
        type=float,
        default=0.45,
        help="Maximum allowed drop in relation evidence vs baseline for control pairs",
    )
    args = parser.parse_args()

    run_prefix = make_run_prefix()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    if not list(results_dir.glob("*.pkl")):
        raise FileNotFoundError(f"No *.pkl extraction results in: {results_dir}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(f"tools/preprocess_book/_v2/runs/bookgraph_v2_{run_prefix}.pkl")
    )
    logs_path = (
        Path(args.logs_path)
        if args.logs_path
        else Path(f"tools/preprocess_book/_v2/runs/v2_decisions_{run_prefix}.jsonl")
    )
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite-output to replace."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = VerificationPipelineV2(
        results_dir=results_dir,
        output_path=output_path,
        logs_path=logs_path,
        top_k=args.top_k,
        llm_type=args.llm_type,
        llm_enabled=not args.disable_llm_judge,
        max_files=args.max_files if args.max_files > 0 else None,
        enable_bridge_merge=not args.disable_bridge_merge,
        judge_concurrency=args.judge_concurrency,
    )
    graph = pipeline.run()

    quality_report = None
    baseline_graph = None
    if args.baseline_graph:
        baseline_path = Path(args.baseline_graph)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline graph not found: {baseline_path}")
        baseline_graph = FileManager.load_pickle(baseline_path)

    if baseline_graph is not None or args.enforce_merge_gates:
        quality_report = evaluate_merge_quality(
            graph=graph,
            baseline_graph=baseline_graph,
            max_relation_drop_ratio=args.max_relation_drop_ratio,
        )
        quality_report_path = (
            Path(args.quality_report_path)
            if args.quality_report_path
            else logs_path.with_name(f"{logs_path.stem}_quality.json")
        )
        quality_report_path.parent.mkdir(parents=True, exist_ok=True)
        quality_report_path.write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Merge quality report saved to %s", quality_report_path)
        if quality_report["warnings"]:
            logger.warning("Merge quality warnings: %s", quality_report["warnings"])
        if args.enforce_merge_gates and not quality_report["passed"]:
            raise RuntimeError(
                "Merge quality gates failed: " + "; ".join(quality_report["violations"])
            )

    logger.info(
        "Done. nodes=%s relations=%s output=%s logs=%s",
        len(graph.nodes.nodes),
        len(graph.relationships.relationships),
        output_path,
        logs_path,
    )


if __name__ == "__main__":
    main()
