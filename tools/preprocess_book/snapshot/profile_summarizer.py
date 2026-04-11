"""Generate short (1-2 sentence) profile summaries for merged entities."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools.preprocess_book.utils.llm_factory import create_llm


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _fallback_summary(entity: Dict[str, Any]) -> str:
    name = str(entity.get("main_name", "")).strip() or "Персонаж"
    classification = str(entity.get("classification", "")).strip() or "сущность"
    actions = entity.get("actions_by_chapter") or []
    snippets: List[str] = []
    for item in actions[:2]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("summary", "")).strip()
        if text:
            snippets.append(text)
    tail = " ".join(snippets).strip()
    if tail:
        return f"{name} — {classification}. {tail[:220]}".strip()
    return f"{name} — {classification}.".strip()


def _normalize_summary(text: str, fallback: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        raw = fallback
    # Keep compact: at most first 2 sentence-like chunks.
    parts = re.split(r"(?<=[.!?])\s+", raw)
    short = " ".join(parts[:2]).strip()
    return short[:320] if short else fallback[:320]


async def generate_entity_profiles(
    *,
    snapshot_dir: Path,
    llm_type: str = "deepseek",
    model: str | None = None,
    concurrency: int = 20,
) -> Dict[str, int]:
    entities_path = snapshot_dir / "entities_merged.jsonl"
    index_path = snapshot_dir / "entity_chapter_index.jsonl"

    entities = _read_jsonl(entities_path)
    index_rows = _read_jsonl(index_path)
    if not entities:
        return {"entities_updated": 0, "index_updated": 0}

    llm = create_llm(llm_type=llm_type, model=model, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты делаешь краткие фактологические профили сущностей для детерминированного поиска. "
                "Пиши строго на русском, 1-2 предложения, без выдумок.",
            ),
            (
                "user",
                "Сущность: {name}\n"
                "Класс: {classification}\n"
                "Алиасы: {aliases}\n"
                "Факты:\n{facts}\n\n"
                "Сделай краткий профиль (1-2 предложения).",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    semaphore = asyncio.Semaphore(max(1, int(concurrency or 1)))

    async def _one(entity: Dict[str, Any]) -> str:
        fallback = _fallback_summary(entity)
        actions = entity.get("actions_by_chapter") or []
        fact_lines: List[str] = []
        for item in actions[:5]:
            if not isinstance(item, dict):
                continue
            cid = item.get("chapter_id")
            summary = str(item.get("summary", "")).strip()
            if summary:
                fact_lines.append(f"[глава={cid}] {summary}")
        facts = "\n".join(fact_lines) or fallback
        payload = {
            "name": str(entity.get("main_name", "")).strip(),
            "classification": str(entity.get("classification", "")).strip(),
            "aliases": ", ".join(entity.get("alt_names") or []),
            "facts": facts,
        }
        try:
            async with semaphore:
                raw = await chain.ainvoke(payload)
            return _normalize_summary(str(raw), fallback)
        except Exception:
            return _normalize_summary("", fallback)

    summaries = await asyncio.gather(*[_one(ent) for ent in entities])
    by_entity_id: Dict[str, str] = {}
    for ent, summary in zip(entities, summaries):
        ent["profile_summary"] = summary
        eid = str(ent.get("entity_id", "")).strip()
        if eid:
            by_entity_id[eid] = summary

    index_updated = 0
    for row in index_rows:
        eid = str(row.get("entity_id", "")).strip()
        if not eid:
            continue
        summary = by_entity_id.get(eid)
        if summary:
            row["profile_summary"] = summary
            index_updated += 1

    _write_jsonl(entities_path, entities)
    if index_rows:
        _write_jsonl(index_path, index_rows)

    return {"entities_updated": len(entities), "index_updated": index_updated}
