from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from sgr_agent_core.base_tool import BaseTool
from src.agent.vector_store.retriever import high_recall_search
from src.config import settings

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class SemanticGraphSearchToolConfig(BaseModel, extra="allow"):
    """Config for semantic retrieval over OpenSearch/vector index."""

    default_k: int = Field(default=int(getattr(settings, "PHASE_A_RECALL_K", 120)), ge=1, description="Default recall size per question")
    max_k: int = Field(default=400, ge=1, description="Hard upper bound for recall size per question")
    max_questions: int = Field(default=20, ge=1, description="Hard upper bound for number of questions in one tool call")


class ChunkHit(BaseModel):
    source_id: int | None = None
    chapter_id: int | None = None
    chunk_id: str | None = None
    text: str = ""


class NodeHit(BaseModel):
    source_id: int | None = None
    chapter_id: int | None = None
    main_name: str | None = None
    classification: str | None = None
    actions: list[str] = Field(default_factory=list)
    actions_text: str = ""


class RelationHit(BaseModel):
    source_id: int | None = None
    chapter_id: int | None = None
    source_main_name: str | None = None
    target_main_name: str | None = None
    relation_type: str | None = None
    actions: list[str] = Field(default_factory=list)
    actions_text: str = ""


class SummaryHit(BaseModel):
    source_id: int | None = None
    chapter_id: int | None = None
    text: str = ""


class SearchCounts(BaseModel):
    chunks: int = 0
    nodes: int = 0
    relations: int = 0
    summaries: int = 0


class SemanticSearchResult(BaseModel):
    question: str
    k: int
    hints: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkHit] = Field(default_factory=list)
    nodes: list[NodeHit] = Field(default_factory=list)
    relations: list[RelationHit] = Field(default_factory=list)
    summaries: list[SummaryHit] = Field(default_factory=list)
    counts: SearchCounts


class SemanticGraphSearchToolResponse(BaseModel):
    status: str
    error: str | None = None
    questions_count: int = 0
    k: int = 0
    results: list[SemanticSearchResult] = Field(default_factory=list)


class SemanticGraphSearchTool(BaseTool):
    """Semantic search for a list of questions with structured output."""

    config_model = SemanticGraphSearchToolConfig

    reasoning: str = Field(description="Why semantic retrieval is needed now")
    questions: list[str] = Field(description="Questions to search semantically", min_length=1, max_length=50)
    k: int | None = Field(default=None, ge=1, description="Recall depth per question (optional)")

    @staticmethod
    def _source_id(meta: dict[str, Any]) -> int | None:
        sid = meta.get("source_id")
        if isinstance(sid, int):
            return sid
        cid = meta.get("chapter_id")
        if isinstance(cid, int):
            return cid
        return None

    @staticmethod
    def _extract_actions_from_text(text: str) -> list[str]:
        out: list[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if line.startswith("- "):
                out.append(line[2:].strip())
        return out

    @staticmethod
    def _normalize_doc(doc: Any) -> tuple[str, dict[str, Any]]:
        text = str(getattr(doc, "page_content", "") or "")
        meta = getattr(doc, "metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = {}
        return text, meta

    def _classify_docs(self, docs: list[Any]) -> tuple[list[ChunkHit], list[NodeHit], list[RelationHit], list[SummaryHit]]:
        chunks: list[ChunkHit] = []
        nodes: list[NodeHit] = []
        relations: list[RelationHit] = []
        summaries: list[SummaryHit] = []

        for doc in docs:
            text, meta = self._normalize_doc(doc)
            source = str(meta.get("source", "")).lower()
            sid = self._source_id(meta)

            if source == "book":
                chunks.append(ChunkHit(source_id=sid, chapter_id=meta.get("chapter_id"), chunk_id=meta.get("chunk_id"), text=text))
                continue

            if source == "nodes":
                nodes.append(
                    NodeHit(
                        source_id=sid,
                        chapter_id=meta.get("chapter_id"),
                        main_name=meta.get("entity_name") or meta.get("name"),
                        classification=meta.get("entity_classification"),
                        actions=self._extract_actions_from_text(text),
                        actions_text=text,
                    )
                )
                continue

            if source == "relations":
                relations.append(
                    RelationHit(
                        source_id=sid,
                        chapter_id=meta.get("chapter_id"),
                        source_main_name=meta.get("relation_object_1"),
                        target_main_name=meta.get("relation_object_2"),
                        relation_type=meta.get("relation_type"),
                        actions=self._extract_actions_from_text(text),
                        actions_text=text,
                    )
                )
                continue

            if source == "summary":
                summaries.append(SummaryHit(source_id=sid, chapter_id=meta.get("chapter_id"), text=text))

        return chunks, nodes, relations, summaries

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        cfg = SemanticGraphSearchToolConfig(**kwargs)

        questions = [str(q).strip() for q in self.questions if str(q).strip()]
        if not questions:
            return SemanticGraphSearchToolResponse(status="error", error="questions is empty after normalization", results=[]).model_dump_json(
                ensure_ascii=False
            )
        if len(questions) > cfg.max_questions:
            questions = questions[: cfg.max_questions]

        k_value = int(self.k or cfg.default_k)
        k_value = max(1, min(k_value, int(cfg.max_k)))

        results: list[SemanticSearchResult] = []
        for q in questions:
            phase = high_recall_search(query=q, k=k_value)
            docs = phase.get("documents") or []
            chunks, nodes, relations, summaries = self._classify_docs(docs)
            results.append(
                SemanticSearchResult(
                    question=q,
                    k=k_value,
                    hints=phase.get("hints") or {},
                    chunks=chunks,
                    nodes=nodes,
                    relations=relations,
                    summaries=summaries,
                    counts=SearchCounts(chunks=len(chunks), nodes=len(nodes), relations=len(relations), summaries=len(summaries)),
                )
            )

        if context.custom_context is None:
            context.custom_context = {}
        context.custom_context["semantic_graph_search_last_query"] = {
            "questions": questions,
            "k": k_value,
            "results": [r.model_dump(mode="json") for r in results],
        }

        return SemanticGraphSearchToolResponse(status="ok", questions_count=len(questions), k=k_value, results=results).model_dump_json(
            ensure_ascii=False
        )
