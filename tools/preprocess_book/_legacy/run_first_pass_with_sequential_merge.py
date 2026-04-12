#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser

from src.utils.graph_search import AllBookNodes, AllBooksEdges
from tools.preprocess_book.make_graph.extraction_service import ExtractionService
from tools.preprocess_book._legacy.make_graph_classic.node_processor import NodeProcessor
from tools.preprocess_book._legacy.make_graph_classic.relation_processor import RelationProcessor
from tools.preprocess_book.make_graph.verification_service import VerificationService
from tools.preprocess_book.prompts.extract_names import ExtractNames
from tools.preprocess_book.prompts.extract_relations import ExtractRelations
from tools.preprocess_book.prompts.verificator import Verification
from tools.preprocess_book.utils.llm_factory import create_llm
from tools.preprocess_book.utils.text_loader import load_book, split_book


async def run(args) -> Path:
    llm = create_llm(llm_type=args.llm_type)
    extractor = ExtractNames(llm=llm)
    extractor_relations = ExtractRelations(llm=llm)
    verificator = Verification(llm=llm, parser=JsonOutputParser())
    verification_service = VerificationService(verificator)
    node_processor = NodeProcessor(verification_service)
    relation_processor = RelationProcessor()
    extraction_service = ExtractionService(
        extractor=extractor,
        extractor_relations=extractor_relations,
    )

    chapters = split_book(load_book(Path(args.book_path)))
    chapters = chapters[: args.max_chapters]

    prompts = []
    source_ids = []
    for doc in chapters:
        sid = int(doc.metadata["source_id"])
        source_ids.append((sid,))
        prompts.append(extractor.make_user_prompt(text=doc.page_content, source_id=sid))

    raw_results = await extractor.batch(prompts, concurrency=args.concurrency)

    all_nodes = AllBookNodes()
    all_rels = AllBooksEdges(relationships={})
    per_source_rows = []

    for sid_tuple, raw_payload in zip(source_ids, raw_results):
        normalized = extraction_service._normalize_result(raw_payload, sid_tuple)
        all_nodes = node_processor.process_input(
            input_data=normalized.get("nodes") or [],
            rel_inputs=normalized.get("relations") or [],
            source_id=sid_tuple,
            all_book_nodes=all_nodes,
        )
        all_rels = relation_processor.process_relations(
            all_book_nodes=all_nodes,
            rel_inputs=normalized.get("relations") or [],
            rel_graphs=all_rels,
            source_id=sid_tuple,
        )

        sid = sid_tuple[0]
        per_source_rows.append(
            {"record_type": "summary", "source_id": sid, "summarization": normalized.get("summarization", "")}
        )
    class_mapping = {
        "EntityClassification.PERSON": "РїРµСЂСЃРѕРЅР°Р¶",
        "EntityClassification.PLACE": "РјРµСЃС‚Рѕ",
        "EntityClassification.ORG": "РѕСЂРіР°РЅРёР·Р°С†РёСЏ",
        "EntityClassification.TERM": "С‚РµСЂРјРёРЅ",
        "EntityClassification.FORCE": "СЃРёР»Р° РїСЂРёСЂРѕРґС‹",
    }

    idx = 1
    for node in all_nodes.nodes.values():
        actions = []
        for act in node.actions or []:
            sid_value = act.source_id[0] if getattr(act, "source_id", None) else None
            chapter_id = getattr(act, "chapter_id", None)
            if chapter_id is None:
                chapter_id = sid_value
            actions.append(
                {
                    "source_id": sid_value,
                    "chapter_id": chapter_id,
                    "description": act.action,
                    "quotes": list(getattr(act, "quotes", []) or []),
                }
            )

        per_source_rows.append(
            {
                "record_type": "node",
                "idx": idx,
                "main_name": node.main_name,
                "classification": class_mapping.get(node.classification, node.classification),
                "alt_names": node.alt_names,
                "actions": actions,
            }
        )
        idx += 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"firstpass_sequential_merge_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in per_source_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run first-pass extraction and apply sequential merge immediately (previous + current chapters)."
    )
    parser.add_argument("--book-path", required=True, type=str)
    parser.add_argument("--max-chapters", default=5, type=int)
    parser.add_argument("--concurrency", default=40, type=int)
    parser.add_argument("--llm-type", default="deepseek", type=str)
    parser.add_argument("--output-dir", default="test_data/results", type=str)
    args = parser.parse_args()

    out_path = asyncio.run(run(args))
    print(out_path)


if __name__ == "__main__":
    main()

