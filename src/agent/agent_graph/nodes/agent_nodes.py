"""
Agent graph nodes for ReAct-style orchestration.
"""

import logging
from copy import deepcopy
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
        "react_max_iterations": int(getattr(settings, "MAX_N_ITERATIONS", 2)) + 2,
        "route_reason": "",
        "final_ready": False,
        "last_tool_name": None,
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
        "chapter_lookup_if_needed",
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

    if state.get("react_iteration", 0) >= state.get("react_max_iterations", 4):
        state["final_ready"] = True
        state["pending_tool_call"] = {}
        state["route_reason"] = "react iteration limit reached"
        return

    last_tool = state.get("last_tool_name")
    hints = state.get("hints", {}) or {}

    if not state.get("tool_calls"):
        tool_name = "semantic_search"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "k": settings.PHASE_A_RECALL_K,
        }
        reason = "no tool calls yet"
    elif last_tool == "semantic_search":
        tool_name = "deterministic_entity_search"
        tool_input = {
            "query": state["user_question"],
            "hints": hints,
            "top_k": settings.PHASE_B_MAX_ENTITIES,
        }
        reason = "semantic hints collected"
    elif last_tool == "deterministic_entity_search":
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
    elif last_tool == "deterministic_graph_expand":
        chapter_ids = hints.get("chapter_ids", [])
        used_chapter_lookup = any(tc.get("tool_name") == "chapter_lookup" for tc in state.get("tool_calls", []))
        if chapter_ids and not used_chapter_lookup:
            tool_name = "chapter_lookup"
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
    elif last_tool == "chapter_lookup":
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

    state["_last_tool_result"] = result
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
    for key in ["nodes_texts", "rel_texts", "quote_texts", "sums_texts", "entities_text", "relations_text", "chapters_text"]:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            chunks.append(f"[{tool_name}:{key}]\n{value.strip()}")
    if not chunks:
        return
    state.setdefault("context", []).append({tool_name: "\n\n".join(chunks)})


def observe_node(state: AgentState) -> AgentState:
    state = _initialize_default_values(state)

    result = state.pop("_last_tool_result", None)
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
