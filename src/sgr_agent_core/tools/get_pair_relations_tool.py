from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class GetPairRelationsToolConfig(BaseModel, extra="allow"):
    """Config for loading pair relations from snapshot."""

    relations_pair_summaries_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/relations_pair_summaries.jsonl",
        description="Path to JSONL with pair relation bundles",
    )
    max_results: int = Field(default=200, ge=1, description="Safety cap for matched pair rows")


class PairRelationMatch(BaseModel):
    pair_id: str | None = Field(default=None, description="Pair identifier")
    source_main_name: str = Field(default="", description="Source entity main_name")
    target_main_name: str = Field(default="", description="Target entity main_name")
    source_summary: str = Field(default="", description="Source entity profile summary")
    target_summary: str = Field(default="", description="Target entity profile summary")
    relations: list[dict[str, Any]] = Field(default_factory=list, description="Raw relation events list from pair bundle")
    relation_summary: str = Field(default="", description="Summarized relation text")


class GetPairRelationsToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    main_name_1: str = Field(default="", description="First entity main_name")
    main_name_2: str = Field(default="", description="Second entity main_name")
    total_matches: int = Field(default=0, ge=0, description="Matched pair rows count")
    matches: list[PairRelationMatch] = Field(default_factory=list, description="Matched pair relation rows")


class GetPairRelationsTool(BaseTool):
    """Return full relations between two entities by main_name (both directions).

    Input:
    - main_name_1
    - main_name_2

    Match rule:
    - source_main_name == main_name_1 and target_main_name == main_name_2
    - OR source_main_name == main_name_2 and target_main_name == main_name_1

    Output rows contain:
    - pair_id
    - source_main_name
    - target_main_name
    - source_summary
    - target_summary
    - relations (full list from pair bundle)
    - relation_summary
    """

    config_model = GetPairRelationsToolConfig

    reasoning: str = Field(description="Why pair-level relation details are needed")
    main_name_1: str = Field(description="First entity main_name")
    main_name_2: str = Field(description="Second entity main_name")

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        return Path.cwd() / path

    @staticmethod
    def _norm(value: str) -> str:
        # Fold yo->e to reduce spelling variance in Russian aliases.
        return str(value or "").strip().lower().replace("ё", "е")

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
        cfg = GetPairRelationsToolConfig(**kwargs)
        path = self._resolve_path(cfg.relations_pair_summaries_path)
        if not path.exists():
            return GetPairRelationsToolResponse(
                status="error",
                error=f"relations pair file not found: {path}",
                total_matches=0,
                matches=[],
            ).model_dump_json(ensure_ascii=False)

        a = self._norm(self.main_name_1)
        b = self._norm(self.main_name_2)
        if not a or not b:
            return GetPairRelationsToolResponse(
                status="error",
                error="main_name_1/main_name_2 must be non-empty",
                main_name_1=self.main_name_1,
                main_name_2=self.main_name_2,
                total_matches=0,
                matches=[],
            ).model_dump_json(ensure_ascii=False)

        rows = self._load_jsonl(path)
        out: list[PairRelationMatch] = []
        for row in rows:
            src_obj = row.get("source_entity") or {}
            dst_obj = row.get("target_entity") or {}
            src = str(src_obj.get("main_name", "")).strip()
            dst = str(dst_obj.get("main_name", "")).strip()
            src_n = self._norm(src)
            dst_n = self._norm(dst)

            direct = src_n == a and dst_n == b
            reverse = src_n == b and dst_n == a
            if not (direct or reverse):
                continue

            summary_obj = row.get("summary") if isinstance(row.get("summary"), dict) else {}
            out.append(
                PairRelationMatch(
                    pair_id=row.get("pair_id"),
                    source_main_name=src,
                    target_main_name=dst,
                    source_summary=str(src_obj.get("profile_summary", "") or ""),
                    target_summary=str(dst_obj.get("profile_summary", "") or ""),
                    relations=row.get("relations") or [],
                    relation_summary=str(summary_obj.get("relation_summary", "") or ""),
                )
            )
            if len(out) >= cfg.max_results:
                break

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["pair_relations_last_query"] = {
            "main_name_1": self.main_name_1,
            "main_name_2": self.main_name_2,
            "total_matches": len(out),
            "matches": [m.model_dump(mode="json") for m in out],
        }

        return GetPairRelationsToolResponse(
            status="ok",
            main_name_1=self.main_name_1,
            main_name_2=self.main_name_2,
            total_matches=len(out),
            matches=out,
        ).model_dump_json(ensure_ascii=False)
