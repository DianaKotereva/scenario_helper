from langgraph.graph import END, START, StateGraph
from src.agent.agent_graph.nodes.agent_nodes import (
    cond_edge_reasoner,
    final_answer_node,
    reasoning_node,
    search_node,
)
from src.agent.agent_graph.states.agent_state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("search_node", search_node)
workflow.add_node("final_answer_node", final_answer_node)

workflow.add_edge(START, "reasoning_node")
workflow.add_conditional_edges(
    "reasoning_node", cond_edge_reasoner, ["search_node", "final_answer_node"]
)
workflow.add_edge("search_node", "reasoning_node")
workflow.add_edge("final_answer_node", END)

app = workflow.compile()
