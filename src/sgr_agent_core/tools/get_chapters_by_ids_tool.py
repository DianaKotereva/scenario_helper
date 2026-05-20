from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class GetChaptersByIdsToolConfig(BaseModel, extra="allow"):
    """Config for chapter lookup from local snapshot."""

    chapters_file_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/chapters.jsonl",
        description="Path to chapters JSONL snapshot file",
    )
    max_chapters: int = Field(default=50, ge=1, description="Safety limit on returned chapters")
    max_text_chars_per_chapter: int = Field(
        default=20000,
        ge=200,
        description="Maximum text characters per chapter in response",
    )


class ChapterPayload(BaseModel):
    chapter_id: int = Field(description="Chapter id")
    source_id: int = Field(description="Source id")
    title: str = Field(default="", description="Chapter title")
    text: str = Field(default="", description="Chapter text (possibly truncated)")


class GetChaptersByIdsToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    requested_ids: list[int] = Field(default_factory=list, description="All requested chapter ids")
    returned_ids: list[int] = Field(default_factory=list, description="Chapter ids successfully returned")
    missing_ids: list[int] = Field(default_factory=list, description="Requested ids that were not found")
    found_count: int = Field(default=0, ge=0, description="Number of found chapters")
    chapters: list[ChapterPayload] = Field(default_factory=list, description="Returned chapters")


class GetChaptersByIdsTool(BaseTool):
    """Return chapter texts for requested chapter numbers (source_id/chapter_id).

    Supports two input modes:
    - explicit ids: chapter_ids=[...]
    - range mode: chapter_id_start + chapter_id_end (inclusive)
    """

    config_model = GetChaptersByIdsToolConfig

    reasoning: str = Field(description="Why these chapters are needed")
    chapter_ids: list[int] | None = Field(default=None, description="Chapter/source ids to fetch", max_length=200)
    chapter_id_start: int | None = Field(default=None, description="Start chapter/source id for range mode (inclusive)")
    chapter_id_end: int | None = Field(default=None, description="End chapter/source id for range mode (inclusive)")

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        return Path.cwd() / path

    @staticmethod
    def _load_chapter_index(path: Path) -> dict[int, dict]:
        idx: dict[int, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                chapter_id = row.get("chapter_id")
                source_id = row.get("source_id")
                key = chapter_id if isinstance(chapter_id, int) else source_id
                if isinstance(key, int):
                    idx[key] = row
        return idx

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        cfg = GetChaptersByIdsToolConfig(**kwargs)
        chapters_path = self._resolve_path(cfg.chapters_file_path)
        if not chapters_path.exists():
            return GetChaptersByIdsToolResponse(
                status="error",
                error=f"chapters file not found: {chapters_path}",
                requested_ids=self.chapter_ids or [],
                found_count=0,
                chapters=[],
            ).model_dump_json(ensure_ascii=False)

        index = self._load_chapter_index(chapters_path)

        range_ids: list[int] = []
        if self.chapter_id_start is not None or self.chapter_id_end is not None:
            if self.chapter_id_start is None or self.chapter_id_end is None:
                return GetChaptersByIdsToolResponse(
                    status="error",
                    error="Both chapter_id_start and chapter_id_end must be set for range mode",
                    requested_ids=self.chapter_ids or [],
                    found_count=0,
                    chapters=[],
                ).model_dump_json(ensure_ascii=False)
            start = int(self.chapter_id_start)
            end = int(self.chapter_id_end)
            if end < start:
                return GetChaptersByIdsToolResponse(
                    status="error",
                    error="chapter_id_end must be >= chapter_id_start",
                    requested_ids=self.chapter_ids or [],
                    found_count=0,
                    chapters=[],
                ).model_dump_json(ensure_ascii=False)
            range_ids = list(range(start, end + 1))

        explicit_ids = [int(cid) for cid in (self.chapter_ids or []) if isinstance(cid, int)]
        all_requested = explicit_ids + range_ids
        if not all_requested:
            return GetChaptersByIdsToolResponse(
                status="error",
                error="Provide chapter_ids or chapter_id_start/chapter_id_end",
                requested_ids=[],
                found_count=0,
                chapters=[],
            ).model_dump_json(ensure_ascii=False)

        unique_ids = []
        seen = set()
        for cid in all_requested:
            if isinstance(cid, int) and cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        selected_ids = unique_ids[: cfg.max_chapters]
        found: list[ChapterPayload] = []
        missing: list[int] = []
        for cid in selected_ids:
            row = index.get(cid)
            if row is None:
                missing.append(cid)
                continue
            text = str(row.get("text", ""))
            chapter_id_value = row.get("chapter_id", cid)
            source_id_value = row.get("source_id", chapter_id_value)
            try:
                chapter_id_int = int(chapter_id_value)
            except Exception:
                chapter_id_int = int(cid)
            try:
                source_id_int = int(source_id_value)
            except Exception:
                source_id_int = int(chapter_id_int)
            found.append(
                ChapterPayload(
                    chapter_id=chapter_id_int,
                    source_id=source_id_int,
                    title=str(row.get("title", "") or ""),
                    text=text[: cfg.max_text_chars_per_chapter],
                )
            )

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["chapters_last_fetch"] = {
            "requested_ids": all_requested,
            "found_count": len(found),
            "missing_ids": missing,
        }

        return GetChaptersByIdsToolResponse(
            status="ok",
            requested_ids=all_requested,
            returned_ids=[c.chapter_id for c in found],
            missing_ids=missing,
            found_count=len(found),
            chapters=found,
        ).model_dump_json(ensure_ascii=False)
