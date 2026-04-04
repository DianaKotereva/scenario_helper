from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.agent.prompts import answer_agent, questions_agent, reasoning_agent, retrieve_agent
from tools.prompt_debug.stage_logger import StageLogger


def _safe_preview(text: Any, limit: int = 400) -> str:
    s = str(text) if text is not None else ""
    return s[:limit]


def _serialize_doc(doc: Document) -> Dict[str, Any]:
    return {
        "source": doc.metadata.get("source"),
        "source_id": doc.metadata.get("source_id"),
        "name": str(doc.metadata.get("name"))[:120] if doc.metadata else None,
        "snippet": _safe_preview(doc.page_content, 260),
    }


class AgentDiagnosticsInstrumentation:
    """Monkey-patch prompt agents to capture stage-level diagnostics."""

    def __init__(self, stage_logger: StageLogger) -> None:
        self.stage_logger = stage_logger
        self.current_case_id: Optional[str] = None
        self._originals: Dict[str, Any] = {}

    def set_case(self, case_id: str) -> None:
        self.current_case_id = case_id

    def _log(self, stage: str, payload: Dict[str, Any]) -> None:
        if not self.current_case_id:
            return
        self.stage_logger.log(self.current_case_id, stage, payload)

    def __enter__(self) -> "AgentDiagnosticsInstrumentation":
        self._patch_reasoning()
        self._patch_question_generation()
        self._patch_retrieval_stack()
        self._patch_final_aggregation()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, (obj, attr, original) in self._originals.items():
            setattr(obj, attr, original)

    def _store_original(self, key: str, obj: Any, attr: str) -> None:
        self._originals[key] = (obj, attr, getattr(obj, attr))

    def _patch_reasoning(self) -> None:
        self._store_original("reasoning_invoke", reasoning_agent, "invoke")
        original = reasoning_agent.invoke

        def wrapped_invoke(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                out = result if isinstance(result, dict) else result.model_dump()
                self._log(
                    "reasoning",
                    {
                        "next_step": out.get("next_step"),
                        "to_collect_preview": _safe_preview(out.get("to_collect"), 240),
                        "reasoning_preview": _safe_preview(out.get("reasoning"), 280),
                    },
                )
                return result
            except Exception as ex:  # noqa: BLE001
                self._log("reasoning_error", {"error": str(ex)})
                raise

        reasoning_agent.invoke = wrapped_invoke

    def _patch_question_generation(self) -> None:
        self._store_original("questions_invoke", questions_agent, "invoke")
        original = questions_agent.invoke

        def wrapped_invoke(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                out = result if isinstance(result, dict) else result.model_dump()
                self._log(
                    "question_generation",
                    {
                        "questions": out.get("questions", []),
                        "reasoning_preview": _safe_preview(out.get("reasoning"), 240),
                    },
                )
                return result
            except Exception as ex:  # noqa: BLE001
                self._log("question_generation_error", {"error": str(ex)})
                raise

        questions_agent.invoke = wrapped_invoke

    def _patch_retrieval_stack(self) -> None:
        retriever_obj = retrieve_agent._retriever
        retriever_cls = retriever_obj.__class__
        self._store_original("retriever_invoke", retriever_cls, "invoke")
        original_retriever_invoke = retriever_cls.invoke

        def wrapped_retriever_invoke(this, *args, **kwargs):
            result = original_retriever_invoke(this, *args, **kwargs)
            docs: List[Document] = result if isinstance(result, list) else []
            by_source: Dict[str, int] = {}
            for doc in docs:
                src = str(doc.metadata.get("source", "unknown"))
                by_source[src] = by_source.get(src, 0) + 1
            self._log(
                "retrieval_search_raw",
                {
                    "query": args[0] if args else kwargs.get("query", ""),
                    "total_docs": len(docs),
                    "by_source": by_source,
                    "top_docs": [_serialize_doc(d) for d in docs[:8]],
                },
            )
            return result

        retriever_cls.invoke = wrapped_retriever_invoke

        self._store_original("retrieve_process_graph", retrieve_agent, "process_graph")
        original_process_graph = retrieve_agent.process_graph

        def wrapped_process_graph(*args, **kwargs):
            graph = original_process_graph(*args, **kwargs)
            self._log(
                "retrieval_graph_processed",
                {
                    "nodes_text_len": len(graph.get("nodes_texts", "")),
                    "rels_text_len": len(graph.get("rel_texts", "")),
                    "nodes_preview": _safe_preview(graph.get("nodes_texts", ""), 260),
                    "rels_preview": _safe_preview(graph.get("rel_texts", ""), 260),
                },
            )
            return graph

        retrieve_agent.process_graph = wrapped_process_graph

        self._store_original("retrieve_make_user_prompt", retrieve_agent, "make_user_prompt")
        original_make_user_prompt = retrieve_agent.make_user_prompt

        def wrapped_make_user_prompt(*args, **kwargs):
            messages = original_make_user_prompt(*args, **kwargs)
            prompt_text = ""
            try:
                prompt_text = messages["messages"][0][1]
            except Exception:
                prompt_text = str(messages)
            self._log(
                "retrieve_prompt_input",
                {
                    "prompt_len": len(prompt_text),
                    "prompt_preview": _safe_preview(prompt_text, 500),
                },
            )
            return messages

        retrieve_agent.make_user_prompt = wrapped_make_user_prompt

        self._store_original("retrieve_invoke", retrieve_agent, "invoke")
        original_retrieve_invoke = retrieve_agent.invoke

        def wrapped_retrieve_invoke(*args, **kwargs):
            try:
                result = original_retrieve_invoke(*args, **kwargs)
                answer = result.get("answer", "") if isinstance(result, dict) else str(result)
                trace = result.get("trace", []) if isinstance(result, dict) else []
                self._log(
                    "retrieve_generation",
                    {
                        "query": kwargs.get("query", ""),
                        "answer_len": len(answer),
                        "answer_preview": _safe_preview(answer, 360),
                    },
                )
                if isinstance(trace, list):
                    for phase_item in trace:
                        if not isinstance(phase_item, dict):
                            continue
                        phase_name = str(phase_item.get("phase", "phase_unknown"))
                        self._log(
                            phase_name,
                            {
                                "query": phase_item.get("query", kwargs.get("query", "")),
                                "input_hints": phase_item.get("input_hints", {}),
                                "selected_items": phase_item.get("selected_items", []),
                                "rejected_items": phase_item.get("rejected_items", []),
                                "reason": phase_item.get("reason", ""),
                                "latency_ms": phase_item.get("latency_ms"),
                                "evidence_refs": phase_item.get("evidence_refs", []),
                            },
                        )
                return result
            except Exception as ex:  # noqa: BLE001
                self._log("retrieve_generation_error", {"error": str(ex)})
                raise

        retrieve_agent.invoke = wrapped_retrieve_invoke

    def _patch_final_aggregation(self) -> None:
        self._store_original("answer_invoke", answer_agent, "invoke")
        original = answer_agent.invoke

        def wrapped_invoke(*args, **kwargs):
            try:
                context = kwargs.get("context", [])
                thoughts = kwargs.get("thoughts", [])
                result = original(*args, **kwargs)
                out = result if isinstance(result, dict) else result.model_dump()
                final_answer = out.get("final_answer", "")
                self._log(
                    "final_aggregation",
                    {
                        "context_items": len(context),
                        "thoughts_items": len(thoughts),
                        "final_answer_len": len(final_answer),
                        "final_answer_preview": _safe_preview(final_answer, 380),
                    },
                )
                return result
            except Exception as ex:  # noqa: BLE001
                self._log("final_aggregation_error", {"error": str(ex)})
                raise

        answer_agent.invoke = wrapped_invoke
