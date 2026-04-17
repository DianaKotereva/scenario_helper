from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    user_question: str
    thoughts: List[str]
    context: List[Dict[str, Any]]
    questions: List[str]
    to_collect: str
    final_answer: str
    next_step: str
    max_n_iterations: int
    n_iteration: int
    stop: bool
    retrieval_trace: List[Dict[str, Any]]
    # ReAct state (P0)
    plan: List[str]
    pending_tool_call: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, Any]]
    hints: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    react_iteration: int
    react_max_iterations: int
    route_reason: str
    final_ready: bool
    last_tool_name: Optional[str]
    last_tool_result: Dict[str, Any]
