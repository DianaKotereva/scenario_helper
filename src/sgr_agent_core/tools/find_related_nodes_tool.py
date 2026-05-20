from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class FindRelatedNodesToolConfig(BaseModel, extra="allow"):
    """Config for searching related nodes in graph snapshots."""

    brief_nodes_file_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/entities_merged_brief.jsonl",
        description="Path to compact nodes JSONL used for shallow search",
    )
    full_nodes_file_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/entities_merged.jsonl",
        description="Path to full nodes JSONL used for deep search",
    )
    max_results: int = Field(default=200, ge=1, description="Maximum matched nodes returned")


class RelatedNodeMatch(BaseModel):
    entity_id: str | None = Field(default=None, description="Entity id")
    main_name: str | None = Field(default=None, description="Canonical name")
    alt_names: list[str] = Field(default_factory=list, description="Alternative names")
    classification: str | None = Field(default=None, description="Entity classification")
    profile_summary: str | None = Field(default=None, description="Entity brief summary")
    matched_terms: list[str] = Field(default_factory=list, description="Normalized terms matched in blob")
    match_score: int = Field(default=0, ge=0, description="Number of matched terms")


class FindRelatedNodesToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    search_method: Literal["shallow", "deep"] = Field(description="Search method used")
    query_terms: list[str] = Field(default_factory=list, description="Input terms after trim")
    normalized_terms: list[str] = Field(default_factory=list, description="Normalized tokens used for matching")
    total_matches: int = Field(default=0, ge=0, description="Number of returned matches")
    matches: list[RelatedNodeMatch] = Field(default_factory=list, description="Matched entities")


class FindRelatedNodesTool(BaseTool):
    """Find nodes related to provided name(s)/word(s).

    Input:
    - query_terms: list of names/words
    - search_method:
      - shallow: search by main_name + alt_names
      - deep: search by main_name + alt_names (same matching surface, different source file)
    """

    config_model = FindRelatedNodesToolConfig

    reasoning: str = Field(description="Why this node search is needed")
    query_terms: list[str] = Field(description="Names/keywords to search", min_length=1, max_length=30)
    search_method: Literal["shallow", "deep"] = Field(
        default="shallow",
        description="Search depth: shallow or deep",
    )

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        return Path.cwd() / path

    @staticmethod
    def _norm_tokens(values: list[str]) -> list[str]:
        out = []
        for v in values:
            for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", str(v).lower()):
                if len(t) >= 2:
                    out.append(t)
        seen = set()
        uniq = []
        for t in out:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq

    @staticmethod
    def _build_shallow_blob(row: dict) -> str:
        return " ".join(
            [
                str(row.get("main_name", "")),
                " ".join(str(x) for x in (row.get("alt_names") or [])),
            ]
        ).lower()

    @staticmethod
    def _build_deep_blob(row: dict) -> str:
        return " ".join(
            [
                str(row.get("main_name", "")),
                " ".join(str(x) for x in (row.get("alt_names") or [])),
            ]
        ).lower()

    def _search_rows(self, rows: list[dict], tokens: list[str], deep: bool, max_results: int) -> list[dict]:
        scored = []
        for row in rows:
            blob = self._build_deep_blob(row) if deep else self._build_shallow_blob(row)
            matched = [t for t in tokens if t in blob]
            if not matched:
                continue
            scored.append(
                {
                    "entity_id": row.get("entity_id"),
                    "main_name": row.get("main_name"),
                    "alt_names": row.get("alt_names", []),
                    "classification": row.get("classification"),
                    "profile_summary": row.get("profile_summary"),
                    "matched_terms": matched,
                    "match_score": len(matched),
                }
            )
        scored.sort(key=lambda x: (-x["match_score"], str(x.get("main_name", ""))))
        return scored[:max_results]

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return rows

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        cfg = FindRelatedNodesToolConfig(**kwargs)
        terms = [str(x).strip() for x in self.query_terms if str(x).strip()]
        tokens = self._norm_tokens(terms)
        if not tokens:
            return FindRelatedNodesToolResponse(
                status="error",
                error="query_terms is empty after normalization",
                search_method=self.search_method,
                total_matches=0,
                matches=[],
            ).model_dump_json(ensure_ascii=False)

        deep = self.search_method == "deep"
        path = self._resolve_path(cfg.full_nodes_file_path if deep else cfg.brief_nodes_file_path)
        if not path.exists():
            return FindRelatedNodesToolResponse(
                status="error",
                error=f"nodes file not found: {path}",
                search_method=self.search_method,
                total_matches=0,
                matches=[],
            ).model_dump_json(ensure_ascii=False)

        rows = self._load_jsonl(path)
        raw_matches = self._search_rows(rows=rows, tokens=tokens, deep=deep, max_results=cfg.max_results)
        matches = [RelatedNodeMatch(**m) for m in raw_matches]

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["related_nodes_last_query"] = {
            "query_terms": terms,
            "search_method": self.search_method,
            "matches": [m.model_dump(mode="json") for m in matches],
        }

        return FindRelatedNodesToolResponse(
            status="ok",
            search_method=self.search_method,
            query_terms=terms,
            normalized_terms=tokens,
            total_matches=len(matches),
            matches=matches,
        ).model_dump_json(ensure_ascii=False)
