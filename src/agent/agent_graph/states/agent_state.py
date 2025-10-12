from typing import TypedDict, List, Dict, Any

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