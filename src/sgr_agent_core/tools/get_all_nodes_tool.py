from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class GetAllNodesToolConfig(BaseModel, extra="allow"):
    """Config for loading graph nodes from local snapshot."""

    nodes_file_path: str = Field(
        default="processed_data/final_snapshot_fullbook_verify_20260502/entities_merged_brief.jsonl",
        description="Path to JSONL file with graph nodes",
    )
    max_nodes: int = Field(
        default=5000,
        ge=1,
        description="Safety limit on number of nodes to load",
    )


class NodeBrief(BaseModel):
    entity_id: str | None = Field(default=None, description="Entity id")
    main_name: str | None = Field(default=None, description="Canonical entity name")
    alt_names: list[str] = Field(default_factory=list, description="Alternative names")
    classification: str | None = Field(default=None, description="Entity type/classification")
    profile_summary: str | None = Field(default=None, description="Short profile summary")


class GetAllNodesToolResponse(BaseModel):
    status: str = Field(description="ok|error")
    error: str | None = Field(default=None, description="Error details if status=error")
    total_nodes: int = Field(default=0, ge=0, description="Number of returned nodes")
    nodes: list[NodeBrief] = Field(default_factory=list, description="Returned nodes")


class GetAllNodesTool(BaseTool):
    """Load the full list of graph nodes into memory for downstream tool steps.

    Returns JSON with:
    - total_nodes
    - nodes: list of {entity_id, main_name, alt_names, classification, profile_summary}
    """

    config_model = GetAllNodesToolConfig

    reasoning: str = Field(description="Why full node list is needed now")

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        # fallback relative to project root (cwd)
        return Path.cwd() / path

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        cfg = GetAllNodesToolConfig(**kwargs)
        nodes_path = self._resolve_path(cfg.nodes_file_path)
        if not nodes_path.exists():
            return GetAllNodesToolResponse(
                status="error",
                error=f"nodes file not found: {nodes_path}",
                total_nodes=0,
                nodes=[],
            ).model_dump_json(
                ensure_ascii=False
            )

        nodes: list[NodeBrief] = []
        with nodes_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                nodes.append(
                    NodeBrief(
                        entity_id=row.get("entity_id"),
                        main_name=row.get("main_name"),
                        alt_names=row.get("alt_names", []),
                        classification=row.get("classification"),
                        profile_summary=row.get("profile_summary"),
                    )
                )
                if len(nodes) >= cfg.max_nodes:
                    break

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["all_graph_nodes"] = [n.model_dump(mode="json") for n in nodes]

        return GetAllNodesToolResponse(
            status="ok",
            total_nodes=len(nodes),
            nodes=nodes,
        ).model_dump_json(
            ensure_ascii=False,
        )
