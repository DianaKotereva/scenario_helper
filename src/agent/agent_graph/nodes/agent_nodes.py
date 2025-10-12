from copy import deepcopy

import src.config.settings as settings
from src.agent.agent_graph.states.agent_state import AgentState
from src.agent.prompts import (
    answer_agent,
    questions_agent,
    reasoning_agent,
    retrieve_agent,
)

default_values = [
    ("max_n_iterations", settings.MAX_N_ITERATIONS),
    ("n_iteration", 0),
    ("user_question", ""),
    ("thoughts", []),
    ("context", []),
    ("questions", []),
    ("final_answer", ""),
    ("stop", False),
]


def reasoning_node(state: AgentState):
    for key, value in deepcopy(default_values):
        if key not in state:
            state[key] = value

    if state["context"]:
        additional_questions = reasoning_agent.invoke(
            user_question=state["user_question"], context=state.get("context", [])
        )
        state["thoughts"].append(additional_questions["reasoning"])
        state["to_collect"] = additional_questions["to_collect"]
        state["next_step"] = additional_questions["next_step"]
        print(additional_questions)
    else:
        state["next_step"] = "SEARCH"
    state["n_iteration"] = state.get("n_iteration", 0) + 1
    return state


def cond_edge_reasoner(state: AgentState):
    if (
        state["next_step"] == "SEARCH"
        and state.get("n_iteration", 0)
        <= state.get("max_n_iterations", settings.MAX_N_ITERATIONS)
        and not state.get("stop", False)
    ):
        return "search_node"
    else:
        return "final_answer_node"


def search_node(state: AgentState):
    done_queries = [list(i.keys())[0] for i in state["context"]]
    if state["user_question"] not in done_queries:
        questions = [state["user_question"]]
    else:
        questions = questions_agent.invoke(
            user_question=state["user_question"],
            reasoning=state["to_collect"],
            context=state["context"],
        )["questions"]
    if questions:
        for query in questions:
            print(query)
            ans = retrieve_agent.invoke(query=query)
            state["context"].append({query: ans["answer"]})
    else:
        state["stop"] = True
    return state


def final_answer_node(state: AgentState):
    final_answer = answer_agent.invoke(
        user_question=state["user_question"],
        thoughts=state["thoughts"],
        context=state["context"],
    )
    state["final_answer"] = final_answer["final_answer"]
    return state
