"""LangGraph workflow for the ReAct-style agent."""

from langgraph.graph import END, START, StateGraph

from src.agent.agent_graph.nodes.agent_nodes import (
    cond_edge_tool_router,
    final_answer_node,
    observe_node,
    planner_node,
    pre_react_bootstrap_node,
    tool_exec_node,
    tool_router_node,
)
from src.agent.agent_graph.states.agent_state import AgentState

# ReAct flow:
# START -> planner -> router -> (tool_exec -> observe -> router)* -> final -> END
workflow = StateGraph(AgentState)

workflow.add_node("pre_react_bootstrap_node", pre_react_bootstrap_node)
workflow.add_node("planner_node", planner_node)
workflow.add_node("tool_router_node", tool_router_node)
workflow.add_node("tool_exec_node", tool_exec_node)
workflow.add_node("observe_node", observe_node)
workflow.add_node("final_answer_node", final_answer_node)

workflow.add_edge(START, "pre_react_bootstrap_node")
workflow.add_edge("pre_react_bootstrap_node", "planner_node")
workflow.add_edge("planner_node", "tool_router_node")
workflow.add_conditional_edges(
    "tool_router_node",
    cond_edge_tool_router,
    ["tool_exec_node", "final_answer_node"],
)
workflow.add_edge("tool_exec_node", "observe_node")
workflow.add_edge("observe_node", "tool_router_node")
workflow.add_edge("final_answer_node", END)

app = workflow.compile()
