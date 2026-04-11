from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field

from hyperextract.types import AutoGraph, AutoHypergraph
from hyperextract.utils.client import get_client


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "tools" / "preprocess_book" / "_hyperextract_runs"
SEED_INPUT_RUN = RUNS_DIR / "hyperextract_25chap_20260411_002452" / "input"


@dataclass
class ChapterItem:
    source_id: int
    text: str


class GraphNode(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type")
    description: str = Field(description="Entity description")


class GraphEdge(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Relation type")
    description: str = Field(description="Relation description")


class HyperNode(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type")
    description: str = Field(description="Entity description")


class HyperEdge(BaseModel):
    members: List[str] = Field(description="List of entities participating in relation")
    relation_type: str = Field(description="Hyper relation type")
    description: str = Field(description="Hyper relation description")


def _load_chapters_0_24() -> List[ChapterItem]:
    if not SEED_INPUT_RUN.exists():
        raise FileNotFoundError(f"Seed input folder not found: {SEED_INPUT_RUN}")

    items: List[ChapterItem] = []
    for sid in range(25):
        fp = SEED_INPUT_RUN / f"{sid}.txt"
        if not fp.exists():
            raise FileNotFoundError(f"Chapter file missing: {fp}")
        text = fp.read_text(encoding="utf-8")
        items.append(ChapterItem(source_id=sid, text=text))
    return items


def _build_combined_text(chapters: List[ChapterItem]) -> str:
    parts: List[str] = []
    for ch in chapters:
        parts.append(f"[SOURCE_ID={ch.source_id}]\n{ch.text.strip()}\n")
    return "\n".join(parts)


def _run_graph():
    llm, emb = get_client()
    graph = AutoGraph[GraphNode, GraphEdge](
        node_schema=GraphNode,
        edge_schema=GraphEdge,
        node_key_extractor=lambda x: x.name.strip().lower(),
        edge_key_extractor=lambda x: f"{x.source.strip().lower()}|{x.relation_type.strip().lower()}|{x.target.strip().lower()}",
        nodes_in_edge_extractor=lambda x: (x.source.strip().lower(), x.target.strip().lower()),
        llm_client=llm,
        embedder=emb,
        extraction_mode="two_stage",
        chunk_size=6000,
        chunk_overlap=400,
        max_workers=20,
        verbose=False,
    )
    return graph


def _run_hypergraph():
    llm, emb = get_client()
    hyper = AutoHypergraph[HyperNode, HyperEdge](
        node_schema=HyperNode,
        edge_schema=HyperEdge,
        node_key_extractor=lambda x: x.name.strip().lower(),
        edge_key_extractor=lambda x: f"{x.relation_type.strip().lower()}|{'|'.join(sorted(m.strip().lower() for m in x.members))}",
        nodes_in_edge_extractor=lambda x: tuple(m.strip().lower() for m in x.members),
        llm_client=llm,
        embedder=emb,
        extraction_mode="two_stage",
        chunk_size=6000,
        chunk_overlap=400,
        max_workers=20,
        verbose=False,
    )
    return hyper


def _feed_by_chapters_with_retry(obj, chapters: List[ChapterItem], retries: int = 3) -> List[int]:
    failed: List[int] = []
    for ch in chapters:
        payload = f"[SOURCE_ID={ch.source_id}]\n{ch.text}"
        ok = False
        last_err = None
        for _ in range(retries):
            try:
                obj.feed_text(payload)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if not ok:
            failed.append(ch.source_id)
            print(f"[WARN] failed source_id={ch.source_id}: {last_err}")
    return failed


def _chapter_coverage_by_node_match(chapters: List[ChapterItem], node_names: List[str]) -> Dict[int, int]:
    cover: Dict[int, int] = {}
    lowered_nodes = [n.lower() for n in node_names if n.strip()]
    for ch in chapters:
        txt = ch.text.lower()
        count = 0
        for n in lowered_nodes:
            if len(n) < 3:
                continue
            if n in txt:
                count += 1
        if count:
            cover[ch.source_id] = count
    return cover


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"hyperextract_eval_25_{ts}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    chapters = _load_chapters_0_24()
    combined_text = _build_combined_text(chapters)

    (run_dir / "input").mkdir(exist_ok=True)
    for ch in chapters:
        (run_dir / "input" / f"{ch.source_id}.txt").write_text(ch.text, encoding="utf-8")
    (run_dir / "input" / "chapters_0_24_ru.txt").write_text(combined_text, encoding="utf-8")

    # Run chapter-by-chapter to avoid one bad chunk breaking the full run.
    graph = _run_graph()
    graph_failed = _feed_by_chapters_with_retry(graph, chapters, retries=3)

    hyper = _run_hypergraph()
    hyper_failed = _feed_by_chapters_with_retry(hyper, chapters, retries=3)

    graph_dump = run_dir / "graph_dump"
    hyper_dump = run_dir / "hypergraph_dump"
    graph.dump(graph_dump)
    hyper.dump(hyper_dump)

    graph_data = graph.data.model_dump()
    hyper_data = hyper.data.model_dump()

    (run_dir / "graph_data.json").write_text(
        json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "hypergraph_data.json").write_text(
        json.dumps(hyper_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    graph_node_names = [n["name"] for n in graph_data.get("nodes", [])]
    hyper_node_names = [n["name"] for n in hyper_data.get("nodes", [])]
    graph_cov = _chapter_coverage_by_node_match(chapters, graph_node_names)
    hyper_cov = _chapter_coverage_by_node_match(chapters, hyper_node_names)

    summary = {
        "run_id": run_id,
        "source_ids": list(range(25)),
        "graph": {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
            "covered_chapters": sorted(graph_cov.keys()),
            "covered_chapters_count": len(graph_cov),
            "failed_chapters": graph_failed,
        },
        "hypergraph": {
            "nodes": len(hyper_data.get("nodes", [])),
            "edges": len(hyper_data.get("edges", [])),
            "covered_chapters": sorted(hyper_cov.keys()),
            "covered_chapters_count": len(hyper_cov),
            "failed_chapters": hyper_failed,
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_id": run_id,
        "seed_input_folder": str(SEED_INPUT_RUN),
        "chapters": [asdict(ch) | {"text": None} for ch in chapters],
        "output": {
            "graph_data": str(run_dir / "graph_data.json"),
            "hypergraph_data": str(run_dir / "hypergraph_data.json"),
            "graph_dump": str(graph_dump),
            "hypergraph_dump": str(hyper_dump),
            "summary": str(run_dir / "run_summary.json"),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
