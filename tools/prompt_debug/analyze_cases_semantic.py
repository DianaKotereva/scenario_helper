#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import json_repair

from src.llm_core.llm_core import llm


SYSTEM_PROMPT = """
Ты аналитик качества RAG-агента.
Для каждого кейса определи:
1) Какие значимые факты есть в извлеченном контексте (retrieval/graph).
2) Какие из них реально попали в финальный ответ.
3) Какие факты из expected отсутствуют в финальном ответе.
4) На каком этапе произошла потеря каждого отсутствующего факта:
   - retrieval_missing (не найдено в retrieved)
   - retrieve_generation_loss (было в retrieved, но не вошло в промежуточный ответ)
   - final_aggregation_loss (было в context/retrieve_answer, но пропало в финале)
   - ambiguous

Верни только JSON:
{
  "case_id": str,
  "question": str,
  "found_in_retrieval": [str],
  "reflected_in_final_answer": [str],
  "missing_vs_expected": [
    {
      "fact": str,
      "lost_stage": str,
      "why": str
    }
  ],
  "overall_primary_problem_stage": str,
  "summary": str
}
"""


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    try:
        return json_repair.loads(text)
    except Exception:  # noqa: BLE001
        return {}


def _compact_case(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    case = case_payload.get("case", {})
    events = case_payload.get("events", [])

    retrieval_docs: List[Dict[str, Any]] = []
    graph_bits: List[Dict[str, Any]] = []
    retrieve_answers: List[str] = []

    for event in events:
        stage = event.get("stage")
        payload = event.get("payload", {})
        if stage == "retrieval_search_raw":
            retrieval_docs.extend(payload.get("top_docs", [])[:6])
        elif stage == "retrieval_graph_processed":
            graph_bits.append(
                {
                    "nodes_preview": payload.get("nodes_preview", ""),
                    "rels_preview": payload.get("rels_preview", ""),
                }
            )
        elif stage == "retrieve_generation":
            retrieve_answers.append(str(payload.get("answer_preview", "")))

    return {
        "case_id": case.get("case_id"),
        "row_id": case.get("row_id"),
        "question": case.get("question", ""),
        "expected": case.get("expected", ""),
        "actual": case.get("actual", ""),
        "retrieval_top_docs": retrieval_docs,
        "graph_previews": graph_bits,
        "retrieve_answer_previews": retrieve_answers,
    }


def analyze_case(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = _compact_case(case_payload)
    user_prompt = "Проанализируй кейс:\n" + json.dumps(compact, ensure_ascii=False)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = response.content if hasattr(response, "content") else str(response)
    parsed = _parse_json(content)
    if not parsed:
        parsed = {
            "case_id": compact["case_id"],
            "question": compact["question"],
            "found_in_retrieval": [],
            "reflected_in_final_answer": [],
            "missing_vs_expected": [],
            "overall_primary_problem_stage": "ambiguous",
            "summary": "LLM judge parse failed.",
        }
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic analysis for prompt_debug case artifacts.")
    parser.add_argument("--run-dir", required=True, help="Path to prompt_debug run directory.")
    parser.add_argument("--row-ids", default="", help="Comma-separated row ids filter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cases_dir = run_dir / "cases"
    selected = set()
    if args.row_ids.strip():
        selected = {int(x) for x in args.row_ids.split(",") if x.strip().isdigit()}

    analyses: List[Dict[str, Any]] = []
    for case_file in sorted(cases_dir.glob("row_*.json")):
        row_id = int(case_file.stem.split("_")[-1])
        if selected and row_id not in selected:
            continue
        payload = json.loads(case_file.read_text(encoding="utf-8"))
        analyses.append(analyze_case(payload))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = run_dir / f"semantic_analysis_{ts}.json"
    md_out = run_dir / f"semantic_analysis_{ts}.md"
    json_out.write_text(json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = ["# Semantic Analysis", ""]
    for item in analyses:
        lines.append(f"## {item.get('case_id', '-')}")
        lines.append(f"- question: {item.get('question', '-')}")
        lines.append(f"- primary_problem_stage: {item.get('overall_primary_problem_stage', '-')}")
        lines.append("- found_in_retrieval:")
        for fact in item.get("found_in_retrieval", []):
            lines.append(f"  - {fact}")
        lines.append("- reflected_in_final_answer:")
        for fact in item.get("reflected_in_final_answer", []):
            lines.append(f"  - {fact}")
        lines.append("- missing_vs_expected:")
        for miss in item.get("missing_vs_expected", []):
            lines.append(
                f"  - fact: {miss.get('fact','-')} | lost_stage: {miss.get('lost_stage','-')} | why: {miss.get('why','-')}"
            )
        lines.append(f"- summary: {item.get('summary', '-')}")
        lines.append("")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved semantic JSON: {json_out}")
    print(f"Saved semantic MD: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
