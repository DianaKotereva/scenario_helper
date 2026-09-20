from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class ResolveEntityNameToolConfig(BaseModel, extra="allow"):
    """Config for entity name resolver."""

    brief_nodes_file_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/entities_merged_brief.jsonl",
        description="Path to entities brief JSONL",
    )
    max_candidates: int = Field(default=10, ge=1, description="Maximum number of candidates in output")


class ResolveCandidate(BaseModel):
    entity_id: str | None = Field(default=None, description="Candidate entity id")
    main_name: str = Field(default="", description="Candidate canonical main_name")
    classification: str | None = Field(default=None, description="Candidate classification")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Match score")
    match_reason: str = Field(default="", description="Reason for the score")


class ResolveResult(BaseModel):
    input_name: str = Field(default="", description="Input name as provided")
    canonical_main_name: str | None = Field(default=None, description="Best canonical main_name")
    canonical_entity_id: str | None = Field(default=None, description="Best canonical entity id")
    candidates: list[ResolveCandidate] = Field(default_factory=list, description="Ranked candidates")


class ResolveEntityNameToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    resolved_count: int = Field(default=0, ge=0, description="Number of resolved input names")
    resolved: list[ResolveResult] = Field(default_factory=list, description="Resolved items")


class ResolveEntityNameTool(BaseTool):
    """Resolve user-provided entity names to canonical graph main_name.

    Input:
    - names: list of names/aliases (e.g. Майтимо/Феанаро/Куруфинвэ)

    Output:
    - resolved: list with canonical_main_name + candidates for each input name
    """

    config_model = ResolveEntityNameToolConfig

    reasoning: str = Field(description="Why entity name resolution is needed")
    names: list[str] = Field(description="Input names to resolve", min_length=1, max_length=100)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        return Path.cwd() / path

    @staticmethod
    def _norm(v: str) -> str:
        # Fold yo->e to reduce spelling variance in Russian aliases.
        return str(v or "").strip().lower().replace("ё", "е")

    @staticmethod
    def _tokens(v: str) -> set[str]:
        return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", ResolveEntityNameTool._norm(v)))

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

    def _score(self, q: str, main_name: str, alt_names: list[str]) -> tuple[float, str]:
        qn = self._norm(q)
        mn = self._norm(main_name)
        alts = [self._norm(a) for a in (alt_names or []) if str(a).strip()]
        if qn == mn:
            return 1.0, "exact_main_name"
        if qn in alts:
            return 0.96, "exact_alt_name"
        if qn in mn:
            return 0.90, "substring_main_name"
        if any(qn in a for a in alts):
            return 0.86, "substring_alt_name"

        qt = self._tokens(qn)
        if not qt:
            return 0.0, "no_tokens"
        mt = self._tokens(mn)
        at = set()
        for a in alts:
            at |= self._tokens(a)
        overlap_main = len(qt & mt) / max(1, len(qt | mt))
        overlap_alt = len(qt & at) / max(1, len(qt | at)) if at else 0.0
        score = max(overlap_main * 0.75, overlap_alt * 0.7)
        reason = "token_overlap_main" if overlap_main >= overlap_alt else "token_overlap_alt"
        return score, reason

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        cfg = ResolveEntityNameToolConfig(**kwargs)
        path = self._resolve_path(cfg.brief_nodes_file_path)
        if not path.exists():
            return ResolveEntityNameToolResponse(
                status="error",
                error=f"entities file not found: {path}",
                resolved=[],
            ).model_dump_json(ensure_ascii=False)

        rows = self._load_jsonl(path)
        entries = []
        for row in rows:
            mn = str(row.get("main_name", "")).strip()
            if not mn:
                continue
            entries.append(
                {
                    "entity_id": row.get("entity_id"),
                    "main_name": mn,
                    "alt_names": row.get("alt_names") or [],
                    "classification": row.get("classification"),
                    "profile_summary": row.get("profile_summary"),
                }
            )

        out: list[ResolveResult] = []
        for raw_name in self.names:
            query = str(raw_name).strip()
            if not query:
                continue
            cands: list[ResolveCandidate] = []
            for e in entries:
                score, reason = self._score(query, e["main_name"], e["alt_names"])
                if score <= 0:
                    continue
                cands.append(
                    ResolveCandidate(
                        entity_id=e["entity_id"],
                        main_name=e["main_name"],
                        classification=e["classification"],
                        score=round(float(score), 4),
                        match_reason=reason,
                    )
                )
            cands.sort(key=lambda x: (-x.score, str(x.main_name)))
            cands = cands[: int(cfg.max_candidates)]

            out.append(
                ResolveResult(
                    input_name=query,
                    canonical_main_name=cands[0].main_name if cands else None,
                    canonical_entity_id=cands[0].entity_id if cands else None,
                    candidates=cands,
                )
            )

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["resolved_entity_names_last_query"] = [r.model_dump(mode="json") for r in out]

        return ResolveEntityNameToolResponse(
            status="ok",
            resolved_count=len(out),
            resolved=out,
        ).model_dump_json(ensure_ascii=False)
