from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class GetEntityRelationsToolConfig(BaseModel, extra="allow"):
    """Config for loading relation summaries and entity briefs from snapshot files."""

    relations_pair_summaries_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/relations_pair_summaries.jsonl",
        description="Path to JSONL with pairwise relation summaries",
    )
    entities_brief_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/entities_merged_brief.jsonl",
        description="Path to JSONL with brief entity profiles",
    )
    max_results: int = Field(default=5000, ge=1, description="Safety limit for returned relation rows")
    max_hops_supported: int = Field(default=1, ge=1, description="Traversal depth is fixed to 1 in this tool")


class EntityRelationItem(BaseModel):
    pair_id: str | None = Field(default=None, description="Pair identifier")
    source_main_name: str = Field(default="", description="Source entity main_name")
    target_main_name: str = Field(default="", description="Target entity main_name")
    source_summary: str = Field(default="", description="Source entity brief summary")
    target_summary: str = Field(default="", description="Target entity brief summary")
    relation_summary: str = Field(default="", description="Summarized relation between source and target")


class GetEntityRelationsToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    requested_main_names: list[str] = Field(default_factory=list, description="Main names requested by user")
    max_hops: int = Field(default=1, ge=1, description="Applied traversal depth")
    relation_types: list[str] = Field(default_factory=list, description="Applied relation type filters")
    source_ids: list[int] = Field(default_factory=list, description="Applied source_id filters")
    total_relations: int = Field(default=0, ge=0, description="Number of returned relation rows")
    relations: list[EntityRelationItem] = Field(default_factory=list, description="Returned relation rows")


class GetEntityRelationsTool(BaseTool):
    """Return all relation pairs where provided entity main_name(s) are involved.

    Input:
    - main_names: list of entity main_name values

    Output rows:
    - pair_id
    - source_main_name
    - target_main_name
    - source_summary
    - target_summary
    - relation_summary
    """

    config_model = GetEntityRelationsToolConfig

    reasoning: str = Field(description="Why these relation neighbors are needed")
    main_names: list[str] = Field(
        description="Entity main_name list to expand by graph relations",
        min_length=1,
        max_length=100,
    )
    relation_types: list[str] | None = Field(default=None, description="Optional filter by relation type list")
    source_ids: list[int] | None = Field(default=None, description="Optional whitelist of source_id values")

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
        cfg = GetEntityRelationsToolConfig(**kwargs)
        rel_path = self._resolve_path(cfg.relations_pair_summaries_path)
        brief_path = self._resolve_path(cfg.entities_brief_path)

        if not rel_path.exists():
            return GetEntityRelationsToolResponse(
                status="error",
                error=f"relations file not found: {rel_path}",
                total_relations=0,
                relations=[],
            ).model_dump_json(ensure_ascii=False)
        if not brief_path.exists():
            return GetEntityRelationsToolResponse(
                status="error",
                error=f"entities brief file not found: {brief_path}",
                total_relations=0,
                relations=[],
            ).model_dump_json(ensure_ascii=False)

        input_names = [str(x).strip() for x in self.main_names if str(x).strip()]
        wanted = {self._norm(x) for x in input_names}
        if not wanted:
            return GetEntityRelationsToolResponse(
                status="error",
                error="main_names is empty after normalization",
                total_relations=0,
                relations=[],
            ).model_dump_json(ensure_ascii=False)

        brief_rows = self._load_jsonl(brief_path)
        summary_by_main: dict[str, str] = {}
        for row in brief_rows:
            main_name = str(row.get("main_name", "")).strip()
            if not main_name:
                continue
            summary_by_main[self._norm(main_name)] = str(row.get("profile_summary", "") or "")

        rel_rows = self._load_jsonl(rel_path)
        relation_type_filter = {self._norm(x) for x in (self.relation_types or []) if str(x).strip()}
        source_id_filter = set(int(x) for x in (self.source_ids or []))

        def row_passes_filters(row: dict) -> bool:
            if relation_type_filter:
                row_types = set()
                for rel in row.get("relations") or []:
                    if isinstance(rel, dict):
                        rt = str(rel.get("type", "")).strip()
                        if rt:
                            row_types.add(self._norm(rt))
                if not row_types.intersection(relation_type_filter):
                    return False
            if source_id_filter:
                found = False
                for rel in row.get("relations") or []:
                    if isinstance(rel, dict) and isinstance(rel.get("source_id"), int):
                        if int(rel["source_id"]) in source_id_filter:
                            found = True
                            break
                if not found:
                    return False
            return True

        rows_by_name: dict[str, list[dict]] = {}
        for row in rel_rows:
            src = str((row.get("source_entity") or {}).get("main_name", "")).strip()
            dst = str((row.get("target_entity") or {}).get("main_name", "")).strip()
            if src:
                rows_by_name.setdefault(self._norm(src), []).append(row)
            if dst:
                rows_by_name.setdefault(self._norm(dst), []).append(row)

        # Fixed one-hop traversal: do not allow model-driven hop depth changes.
        depth = 1
        frontier = set(wanted)
        visited_names: set[str] = set()
        selected_row_ids: set[str] = set()
        selected_rows: list[dict] = []

        for _ in range(depth):
            next_frontier: set[str] = set()
            for name_n in list(frontier):
                if name_n in visited_names:
                    continue
                visited_names.add(name_n)
                for row in rows_by_name.get(name_n, []):
                    pair_id = str(row.get("pair_id", "") or "")
                    if not pair_id or pair_id in selected_row_ids:
                        continue
                    if not row_passes_filters(row):
                        continue
                    selected_row_ids.add(pair_id)
                    selected_rows.append(row)

                    src = str((row.get("source_entity") or {}).get("main_name", "")).strip()
                    dst = str((row.get("target_entity") or {}).get("main_name", "")).strip()
                    if src:
                        next_frontier.add(self._norm(src))
                    if dst:
                        next_frontier.add(self._norm(dst))
            frontier = next_frontier

        out: list[EntityRelationItem] = []
        for row in selected_rows:
            src = str((row.get("source_entity") or {}).get("main_name", "")).strip()
            dst = str((row.get("target_entity") or {}).get("main_name", "")).strip()
            src_n = self._norm(src)
            dst_n = self._norm(dst)

            relation_summary = ""
            summary_obj = row.get("summary")
            if isinstance(summary_obj, dict):
                relation_summary = str(summary_obj.get("relation_summary", "") or "")

            src_summary = summary_by_main.get(src_n) or str((row.get("source_entity") or {}).get("profile_summary", "") or "")
            dst_summary = summary_by_main.get(dst_n) or str((row.get("target_entity") or {}).get("profile_summary", "") or "")

            out.append(
                EntityRelationItem(
                    pair_id=row.get("pair_id"),
                    source_main_name=src,
                    target_main_name=dst,
                    source_summary=src_summary,
                    target_summary=dst_summary,
                    relation_summary=relation_summary,
                )
            )
            if len(out) >= cfg.max_results:
                break

        out.sort(key=lambda x: (str(x.source_main_name), str(x.target_main_name), str(x.pair_id or "")))

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["entity_relations_last_query"] = {
            "main_names": input_names,
            "max_hops": depth,
            "total_relations": len(out),
            "relations": [r.model_dump(mode="json") for r in out],
        }

        return GetEntityRelationsToolResponse(
            status="ok",
            requested_main_names=input_names,
            max_hops=depth,
            relation_types=self.relation_types or [],
            source_ids=sorted(source_id_filter) if source_id_filter else [],
            total_relations=len(out),
            relations=out,
        ).model_dump_json(ensure_ascii=False)
