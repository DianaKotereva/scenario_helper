from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.preprocess_book.prompts.summarize_relations_pair import SummarizeRelationsPair
from tools.preprocess_book.utils.llm_factory import create_llm


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _entity_index(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for e in entities:
        name = str(e.get("main_name", "")).strip()
        if name:
            idx[name] = e
    return idx


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _auto_summary(source: str, target: str, rel_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    relation_types = []
    seen = set()
    for r in rel_rows:
        t = str(r.get("type", "")).strip()
        if t and t not in seen:
            seen.add(t)
            relation_types.append(t)
    evidence_count = len(rel_rows)
    rel_summary = (
        f"Между «{source}» и «{target}» зафиксировано {evidence_count} наблюдений. "
        f"Ключевые типы связей: {', '.join(relation_types[:10]) if relation_types else 'не указано'}."
    )
    return {
        "relation_summary": rel_summary,
        "kinship_summary": "не указано",
        "relation_types": relation_types[:10],
        "confidence": "medium" if evidence_count >= 2 else "low",
    }


def build_relations_pair_summaries(
    snapshot_dir: Path | str,
    output_file: str | Path = "relations_pair_summaries.jsonl",
    use_llm: bool = False,
    llm_type: str = "deepseek",
    llm_model: str | None = None,
    concurrency: int = 16,
    limit_pairs: int = 0,
) -> Dict[str, Any]:
    snapshot_dir = Path(snapshot_dir)
    relations_path = snapshot_dir / "relations.jsonl"
    entities_path = snapshot_dir / "entities_merged.jsonl"
    out_path = Path(output_file)
    if not out_path.is_absolute():
        out_path = snapshot_dir / out_path

    relations = _load_jsonl(relations_path)
    entities = _load_jsonl(entities_path)
    ent_idx = _entity_index(entities)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in relations:
        s = str(r.get("source_entity", "")).strip()
        t = str(r.get("target_entity", "")).strip()
        if not s or not t:
            continue
        key = _pair_key(s, t)
        grouped[key].append(
            {
                "type": r.get("type", ""),
                "description": r.get("description", ""),
                "source_id": r.get("source_id"),
                "chapter_id": r.get("chapter_id"),
            }
        )

    pair_items = list(grouped.items())
    pair_items.sort(key=lambda kv: (kv[0][0], kv[0][1]))
    if limit_pairs > 0:
        pair_items = pair_items[: limit_pairs]

    summarizer = None
    if use_llm:
        llm = create_llm(llm_type=llm_type, model=llm_model, temperature=0)
        summarizer = SummarizeRelationsPair(llm=llm)

    def _build_row(item: Tuple[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[str, Any]:
        (left, right), rel_rows = item
        left_ent = ent_idx.get(left, {"main_name": left, "alt_names": []})
        right_ent = ent_idx.get(right, {"main_name": right, "alt_names": []})

        if summarizer is None:
            summary = _auto_summary(left, right, rel_rows)
            mode = "auto"
        else:
            try:
                parsed = summarizer.invoke(
                    source_entity={
                        "main_name": left_ent.get("main_name", left),
                        "alt_names": left_ent.get("alt_names") or [],
                        "profile_summary": str(left_ent.get("profile_summary", "") or ""),
                    },
                    target_entity={
                        "main_name": right_ent.get("main_name", right),
                        "alt_names": right_ent.get("alt_names") or [],
                        "profile_summary": str(right_ent.get("profile_summary", "") or ""),
                    },
                    relations=rel_rows,
                )
                if hasattr(parsed, "model_dump"):
                    summary = parsed.model_dump(mode="json")
                elif isinstance(parsed, dict):
                    summary = parsed
                else:
                    summary = {
                        "relation_summary": str(parsed),
                        "kinship_summary": "не указано",
                        "relation_types": [],
                        "confidence": "low",
                    }
                mode = "llm"
            except Exception as exc:  # noqa: BLE001
                summary = _auto_summary(left, right, rel_rows)
                summary["relation_summary"] += f" [fallback due to llm_error: {exc}]"
                mode = "fallback_auto"

        return {
            "pair_id": f"{left}::{right}",
            "source_entity": {
                "main_name": left_ent.get("main_name", left),
                "alt_names": left_ent.get("alt_names") or [],
                "profile_summary": str(left_ent.get("profile_summary", "") or ""),
            },
            "target_entity": {
                "main_name": right_ent.get("main_name", right),
                "alt_names": right_ent.get("alt_names") or [],
                "profile_summary": str(right_ent.get("profile_summary", "") or ""),
            },
            "evidence_count": len(rel_rows),
            "relations": rel_rows,
            "summary": summary,
            "summary_mode": mode,
        }

    rows: List[Dict[str, Any]] = []
    if summarizer is None or concurrency <= 1:
        for item in pair_items:
            rows.append(_build_row(item))
    else:
        max_workers = max(1, int(concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_build_row, item): i for i, item in enumerate(pair_items)}
            ordered_rows: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered_rows[idx] = fut.result()
            for i in range(len(pair_items)):
                rows.append(ordered_rows[i])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output_path": str(out_path),
        "pairs": len(pair_items),
        "mode": "llm" if use_llm else "auto",
        "concurrency": int(concurrency),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pair-wise relation summaries from relations.jsonl")
    parser.add_argument(
        "--snapshot-dir",
        default="processed_data/final_snapshot_fullbook_verify_20260502",
        help="Path to snapshot directory",
    )
    parser.add_argument(
        "--output-file",
        default="relations_pair_summaries.jsonl",
        help="Output JSONL file name (inside snapshot dir unless absolute path)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM summarizer prompt for each pair",
    )
    parser.add_argument("--llm-type", default="deepseek", help="deepseek|chatgpt")
    parser.add_argument("--llm-model", default=None, help="Optional model override")
    parser.add_argument("--concurrency", type=int, default=16, help="LLM concurrency for pair summarization")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Optional cap on number of pairs")
    args = parser.parse_args()

    stats = build_relations_pair_summaries(
        snapshot_dir=args.snapshot_dir,
        output_file=args.output_file,
        use_llm=args.use_llm,
        llm_type=args.llm_type,
        llm_model=args.llm_model,
        concurrency=args.concurrency,
        limit_pairs=args.limit_pairs,
    )
    print(f"WROTE {stats['output_path']}")
    print(f"PAIRS {stats['pairs']}")
    print(f"MODE {stats['mode']}")
    print(f"CONCURRENCY {stats['concurrency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
