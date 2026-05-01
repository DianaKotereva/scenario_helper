"""
Agent graph nodes for ReAct-style orchestration.
"""

import logging
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List

import src.config.settings as settings
from src.agent.agent_graph.constants import AgentDefaults, NextStep
from src.agent.agent_graph.states.agent_state import AgentState
from src.agent.prompts import answer_agent, retrieve_agent

logger = logging.getLogger(__name__)


def _merge_unique_ints(existing: Iterable[int], new_values: Iterable[int]) -> List[int]:
    out = {int(v) for v in existing if isinstance(v, int)}
    out.update(int(v) for v in new_values if isinstance(v, int))
    return sorted(out)


def _merge_unique_strs(existing: Iterable[str], new_values: Iterable[str]) -> List[str]:
    out = {str(v) for v in existing if isinstance(v, str) and str(v).strip()}
    out.update(str(v) for v in new_values if isinstance(v, str) and str(v).strip())
    return sorted(out)


def _initialize_default_values(state: AgentState) -> AgentState:
    defaults: Dict[str, Any] = {
        "max_n_iterations": settings.MAX_N_ITERATIONS,
        "n_iteration": AgentDefaults.N_ITERATION,
        "user_question": AgentDefaults.USER_QUESTION,
        "thoughts": deepcopy(AgentDefaults.THOUGHTS),
        "context": deepcopy(AgentDefaults.CONTEXT),
        "questions": deepcopy(AgentDefaults.QUESTIONS),
        "final_answer": AgentDefaults.FINAL_ANSWER,
        "stop": AgentDefaults.STOP,
        "retrieval_trace": [],
        "plan": [],
        "pending_tool_call": {},
        "tool_calls": [],
        "observations": [],
        "hints": {
            "chapter_ids": [],
            "source_ids": [],
            "entity_ids": [],
            "entity_name_tokens": [],
            "entity_names": [],
        },
        "evidence": [],
        "react_iteration": 0,
        # Relax ReAct budget: allow extra exploration before final answer.
        # This keeps chapter fetch guaranteed while allowing additional cycles.
        "react_max_iterations": max(int(getattr(settings, "MAX_N_ITERATIONS", 2)) + 6, 8),
        "route_reason": "",
        "final_ready": False,
        "last_tool_name": None,
        "last_tool_result": {},
        "next_step": NextStep.SEARCH.value,
    }

    for key, value in defaults.items():
        if key not in state:
            state[key] = deepcopy(value) if isinstance(value, (dict, list)) else value
    return state


def _validate_state(state: AgentState) -> None:
    question = str(state.get("user_question", "")).strip()
    if not question:
        raise ValueError("user_question is required in state")
    state["user_question"] = question


def planner_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)
    _validate_state(state)

    state["plan"] = [
        "semantic_search",
        "deterministic_entity_search",
        "deterministic_graph_expand",
        "fulltext_keyword_search",
        "chapter_fetch_if_needed",
        "final_answer",
    ]

    if not state.get("tool_calls"):
        state["pending_tool_call"] = {
            "tool_name": "semantic_search",
            "input": {
                "query": state["user_question"],
                "hints": state.get("hints", {}),
                "k": settings.PHASE_A_RECALL_K,
            },
        }
        state["route_reason"] = "bootstrap React plan with semantic_search"
    return state


def _prepare_next_tool_call(state: AgentState) -> None:
    if state.get("final_ready"):
        state["pending_tool_call"] = {}
        state["route_reason"] = "final_ready already set"
        return

    if state.get("react_iteration", 0) >= state.get("react_max_iterations", 8):
        # Hard stop only if chapter fetch has already happened (or no chapters to fetch).
        hints = state.get("hints", {}) or {}
        tool_calls = state.get("tool_calls", []) or []
        used_tool_names = {str(tc.get("tool_name", "")) for tc in tool_calls}
        chapter_ids = hints.get("chapter_ids", []) or []
        chapter_done = ("chapter_fetch" in used_tool_names) or ("chapter_lookup" in used_tool_names)
        if chapter_done or not chapter_ids:
            state["final_ready"] = True
            state["pending_tool_call"] = {}
            state["route_reason"] = "react iteration limit reached"
            return

    last_tool = state.get("last_tool_name")
    hints = state.get("hints", {}) or {}
    tool_calls = state.get("tool_calls", []) or []
    used_tool_names = {str(tc.get("tool_name", "")) for tc in tool_calls}

    if not tool_calls:
        tool_name = "semantic_search"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "k": settings.PHASE_A_RECALL_K,
        }
        reason = "no tool calls yet"
    elif "deterministic_entity_search" not in used_tool_names:
        tool_name = "deterministic_entity_search"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "top_k": settings.PHASE_B_MAX_ENTITIES,
        }
        reason = "semantic hints collected"
    elif "deterministic_graph_expand" not in used_tool_names:
        tool_name = "deterministic_graph_expand"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "depth": 1,
            "limits": {
                "max_entities": settings.PHASE_B_MAX_ENTITIES,
                "max_relations": settings.PHASE_B_MAX_RELATIONS,
                "max_chapters": settings.PHASE_B_MAX_CHAPTERS,
            },
        }
        reason = "entity candidates available"
    elif "fulltext_keyword_search" not in used_tool_names:
        tool_name = "fulltext_keyword_search"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "max_hits": max(10, int(getattr(settings, "MAX_CHAPTERS_IN_CONTEXT", 10))),
        }
        reason = "coverage safety-net over raw chapter texts"
    elif "chapter_fetch" not in used_tool_names and "chapter_lookup" not in used_tool_names:
        chapter_ids = hints.get("chapter_ids", [])
        if chapter_ids:
            tool_name = "chapter_fetch"
            tool_input = {
                "chapter_ids": chapter_ids,
                "max_chapters": settings.MAX_CHAPTERS_IN_CONTEXT,
            }
            reason = "need chapter grounding"
        else:
            state["final_ready"] = True
            state["pending_tool_call"] = {}
            state["route_reason"] = "graph expansion complete"
            return
    elif last_tool in {"chapter_lookup", "chapter_fetch"}:
        state["final_ready"] = True
        state["pending_tool_call"] = {}
        state["route_reason"] = "chapter grounding complete"
        return
    else:
        state["final_ready"] = True
        state["pending_tool_call"] = {}
        state["route_reason"] = "unknown tool path; stop"
        return

    state["pending_tool_call"] = {"tool_name": tool_name, "input": tool_input}
    state["route_reason"] = reason


def tool_router_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)
    _validate_state(state)

    if not state.get("pending_tool_call"):
        _prepare_next_tool_call(state)

    if state.get("final_ready"):
        state["next_step"] = NextStep.ANSWER.value
    else:
        state["next_step"] = NextStep.SEARCH.value
    return state


def cond_edge_tool_router(state: AgentState) -> str:
    if state.get("final_ready"):
        return "final_answer_node"
    pending = state.get("pending_tool_call") or {}
    if pending.get("tool_name"):
        return "tool_exec_node"
    return "final_answer_node"


def tool_exec_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)
    _validate_state(state)

    pending = state.get("pending_tool_call") or {}
    tool_name = pending.get("tool_name")
    tool_input = pending.get("input") or {}

    if not tool_name:
        state["final_ready"] = True
        state["route_reason"] = "empty pending tool call"
        return state

    try:
        result = retrieve_agent.run_tool(tool_name=tool_name, **tool_input)
    except Exception as exc:  # noqa: BLE001
        logger.error("Tool execution failed for %s: %s", tool_name, exc, exc_info=True)
        result = {
            "tool_name": tool_name,
            "input": tool_input,
            "selected_items": [],
            "rejected_items": [{"reason": str(exc)}],
            "hints": {},
            "context": {},
            "latency_ms": 0.0,
            "error": str(exc),
        }

    state["last_tool_result"] = result
    state["last_tool_name"] = tool_name
    state["react_iteration"] = int(state.get("react_iteration", 0)) + 1
    state["pending_tool_call"] = {}

    state.setdefault("tool_calls", []).append(
        {
            "iteration": state["react_iteration"],
            "tool_name": tool_name,
            "input": tool_input,
            "latency_ms": result.get("latency_ms", 0.0),
            "route_reason": state.get("route_reason", ""),
        }
    )
    return state


def _append_context_from_tool(state: AgentState, tool_name: str, context: Dict[str, Any]) -> None:
    chunks: List[str] = []
    for key in [
        "nodes_texts",
        "rel_texts",
        "quote_texts",
        "sums_texts",
        "entities_text",
        "relations_text",
        "chapters_text",
        "fulltext_hits_text",
    ]:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(f"[{tool_name}:{key}]\n{value.strip()}")
    if not chunks:
        return
    state.setdefault("context", []).append({tool_name: "\n\n".join(chunks)})


def _apply_tool_result_to_state(
    state: AgentState,
    tool_name: str,
    tool_input: Dict[str, Any],
    result: Dict[str, Any],
    route_reason: str,
    iteration: int,
) -> None:
    state["last_tool_name"] = tool_name
    state["last_tool_result"] = result
    state.setdefault("tool_calls", []).append(
        {
            "iteration": iteration,
            "tool_name": tool_name,
            "input": tool_input,
            "latency_ms": result.get("latency_ms", 0.0),
            "route_reason": route_reason,
        }
    )
    observe_node(state)
    state["last_tool_result"] = {}


def pre_react_bootstrap_node(state: AgentState) -> AgentState:
    """
    Mandatory retrieval bootstrap before ReAct:
    1) semantic search with graph extraction
    2) strict keyword fulltext scan
    3) chapter fetch by merged chapter ids
    4) async chapter analysis for dense evidence
    """
    state = _initialize_default_values(state)
    _validate_state(state)
    if state.get("bootstrap_done"):
        return state

    query = state["user_question"]
    hints = state.get("hints", {}) or {}
    semantic_chapter_ids: List[int] = []
    fulltext_chapter_ids: List[int] = []

    bootstrap_plan = [
        (
            "semantic_search",
            {
                "query": query,
                "hints": hints,
                "k": settings.PHASE_A_RECALL_K,
            },
            "bootstrap phase 1/4: semantic graph retrieval",
        ),
        (
            "fulltext_keyword_search",
            {
                "query": query,
                "hints": hints,
                "max_hits": max(20, int(getattr(settings, "MAX_CHAPTERS_IN_CONTEXT", 10)) * 2),
            },
            "bootstrap phase 2/4: strict keyword fulltext retrieval",
        ),
    ]

    iteration = 0
    for tool_name, tool_input, reason in bootstrap_plan:
        iteration += 1
        try:
            result = retrieve_agent.run_tool(tool_name=tool_name, **tool_input)
        except Exception as exc:  # noqa: BLE001
            logger.error("Bootstrap tool failed for %s: %s", tool_name, exc, exc_info=True)
            result = {
                "tool_name": tool_name,
                "input": tool_input,
                "selected_items": [],
                "rejected_items": [{"reason": str(exc)}],
                "hints": {},
                "context": {},
                "latency_ms": 0.0,
                "error": str(exc),
            }
        _apply_tool_result_to_state(
            state=state,
            tool_name=tool_name,
            tool_input=tool_input,
            result=result,
            route_reason=reason,
            iteration=iteration,
        )
        if tool_name == "semantic_search":
            semantic_chapter_ids = [
                int(v) for v in (result.get("hints", {}).get("chapter_ids", []) or []) if isinstance(v, int)
            ]
        if tool_name == "fulltext_keyword_search":
            fulltext_chapter_ids = [
                int(v) for v in (result.get("hints", {}).get("chapter_ids", []) or []) if isinstance(v, int)
            ]
        hints = state.get("hints", {}) or {}

    max_bootstrap_chapters = max(12, int(getattr(settings, "MAX_CHAPTERS_IN_CONTEXT", 10)) * 2)
    merged_chapters: List[int] = []
    for cid in fulltext_chapter_ids + semantic_chapter_ids:
        if cid not in merged_chapters:
            merged_chapters.append(cid)
    chapter_ids = merged_chapters[:max_bootstrap_chapters]
    if not chapter_ids:
        chapter_ids = (hints.get("chapter_ids", []) or [])[:max_bootstrap_chapters]

    chapter_fetch_input = {
        "chapter_ids": chapter_ids,
        "max_chapters": max_bootstrap_chapters,
    }
    iteration += 1
    try:
        chapter_fetch_result = retrieve_agent.run_tool(tool_name="chapter_fetch", **chapter_fetch_input)
    except Exception as exc:  # noqa: BLE001
        logger.error("Bootstrap tool failed for chapter_fetch: %s", exc, exc_info=True)
        chapter_fetch_result = {
            "tool_name": "chapter_fetch",
            "input": chapter_fetch_input,
            "selected_items": [],
            "rejected_items": [{"reason": str(exc)}],
            "hints": {},
            "context": {},
            "latency_ms": 0.0,
            "error": str(exc),
        }
    _apply_tool_result_to_state(
        state=state,
        tool_name="chapter_fetch",
        tool_input=chapter_fetch_input,
        result=chapter_fetch_result,
        route_reason="bootstrap phase 3/4: fetch all candidate chapters",
        iteration=iteration,
    )

    hints = state.get("hints", {}) or {}
    # Analyze chapters asynchronously one-by-one, then aggregate.
    # This keeps per-chapter analysis isolated and avoids one monolithic pass.
    iteration += 1
    async_started = time.perf_counter() if "time" in globals() else None
    analysis_workers = max(1, min(12, len(chapter_ids)))
    chapter_results: List[Dict[str, Any]] = []
    chapter_errors: List[Dict[str, Any]] = []

    def _analyze_single(ch_id: int) -> Dict[str, Any]:
        return retrieve_agent.run_tool(
            tool_name="async_chapter_analysis",
            query=query,
            chapter_ids=[int(ch_id)],
            max_hits_per_chapter=3,
            workers=1,
        )

    try:
        with ThreadPoolExecutor(max_workers=analysis_workers) as pool:
            futures = {pool.submit(_analyze_single, cid): cid for cid in chapter_ids}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    chapter_results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    chapter_errors.append({"chapter_id": cid, "reason": str(exc)})
    except Exception as exc:  # noqa: BLE001
        chapter_errors.append({"reason": str(exc)})

    merged_selected: List[Dict[str, Any]] = []
    merged_chapter_ids: List[int] = []
    merged_blocks: List[str] = []
    for res in chapter_results:
        for item in (res.get("selected_items") or []):
            if isinstance(item, dict):
                merged_selected.append(item)
                cid = item.get("chapter_id")
                if isinstance(cid, int) and cid not in merged_chapter_ids:
                    merged_chapter_ids.append(cid)
        ctx = res.get("context") or {}
        block = ctx.get("chapters_text")
        if isinstance(block, str) and block.strip():
            merged_blocks.append(block.strip())

    async_analysis_result = {
        "tool_name": "async_chapter_analysis",
        "input": {
            "query": query,
            "chapter_ids": chapter_ids,
            "max_hits_per_chapter": 3,
            "workers": analysis_workers,
            "mode": "per_chapter_async",
        },
        "selected_items": merged_selected,
        "rejected_items": chapter_errors,
        "hints": {
            "chapter_ids": merged_chapter_ids,
            "source_ids": merged_chapter_ids,
        },
        "context": {
            "chapters_text": "\n\n***\n\n".join(merged_blocks),
        },
        "reason": "bootstrap phase 4/4: per-chapter async evidence analysis",
        "latency_ms": round(((time.perf_counter() - async_started) * 1000), 2) if async_started else 0.0,
    }
    _apply_tool_result_to_state(
        state=state,
        tool_name="async_chapter_analysis",
        tool_input=async_analysis_result.get("input", {}),
        result=async_analysis_result,
        route_reason="bootstrap phase 4/4: per-chapter async evidence analysis",
        iteration=iteration,
    )

    state["bootstrap_done"] = True
    return state


def observe_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)

    result = state.get("last_tool_result")
    state["last_tool_result"] = {}
    if not isinstance(result, dict):
        return state

    tool_name = str(result.get("tool_name", "unknown"))
    state.setdefault("observations", []).append(result)

    hints = result.get("hints") or {}
    state_hints = state.setdefault("hints", {})
    state_hints["chapter_ids"] = _merge_unique_ints(state_hints.get("chapter_ids", []), hints.get("chapter_ids", []))
    state_hints["source_ids"] = _merge_unique_ints(state_hints.get("source_ids", []), hints.get("source_ids", []))
    state_hints["entity_ids"] = _merge_unique_strs(state_hints.get("entity_ids", []), hints.get("entity_ids", []))
    state_hints["entity_name_tokens"] = _merge_unique_strs(state_hints.get("entity_name_tokens", []), hints.get("entity_name_tokens", []))
    state_hints["entity_names"] = _merge_unique_strs(state_hints.get("entity_names", []), hints.get("entity_names", []))

    context = result.get("context") or {}
    if isinstance(context, dict):
        # source_ids from semantic context
        state_hints["source_ids"] = _merge_unique_ints(state_hints.get("source_ids", []), context.get("source_ids", []))
        _append_context_from_tool(state, tool_name, context)

    selected_items = result.get("selected_items")
    state.setdefault("retrieval_trace", []).append(
        {
            "query": state.get("user_question", ""),
            "phases": [
                {
                    "phase": tool_name,
                    "query": state.get("user_question", ""),
                    "input_hints": result.get("input", {}).get("hints", {}),
                    "selected_items": selected_items if isinstance(selected_items, list) else [],
                    "rejected_items": result.get("rejected_items", []),
                    "reason": state.get("route_reason", ""),
                    "latency_ms": result.get("latency_ms", 0.0),
                }
            ],
        }
    )

    evidence_item = {
        "tool_name": tool_name,
        "selected_count": len(selected_items) if isinstance(selected_items, list) else 0,
        "latency_ms": result.get("latency_ms", 0.0),
    }
    state.setdefault("evidence", []).append(evidence_item)

    return state


def final_answer_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)
    _validate_state(state)

    try:
        result = answer_agent.invoke(
            user_question=state["user_question"],
            thoughts=state.get("thoughts", []),
            context=state.get("context", []),
        )

        from src.agent.prompts.output_models import FinalAnswerOutput

        if isinstance(result, FinalAnswerOutput):
            final_answer = result.final_answer
        elif isinstance(result, dict):
            try:
                final_answer = FinalAnswerOutput(**result).final_answer
            except Exception:
                final_answer = str(result.get("final_answer", "")).strip()
        else:
            final_answer = str(result)

        if final_answer:
            state["final_answer"] = final_answer
        else:
            state["final_answer"] = "?? ??????? ???????????? ????? ?? ?????? ?????????? ?????????."

    except Exception as exc:  # noqa: BLE001
        logger.error("Error in final_answer_node: %s", exc, exc_info=True)
        state["final_answer"] = f"????????? ?????? ??? ???????????? ??????: {exc}"

    state["stop"] = True
    return state


# Backward-compatible wrappers for existing imports/tests.
def reasoning_node(state: AgentState) -> AgentState:
    return planner_node(state)


def search_node(state: AgentState) -> AgentState:
    state = tool_router_node(state)
    if not state.get("final_ready"):
        state = tool_exec_node(state)
        state = observe_node(state)
    return state


def cond_edge_reasoner(state: AgentState) -> str:
    if state.get("final_ready"):
        return "final_answer_node"
    return "search_node"
