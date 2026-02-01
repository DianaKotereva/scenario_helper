"""
Константы для агента.
"""

from enum import Enum


class NextStep(str, Enum):
    """Возможные следующие шаги агента."""
    SEARCH = "SEARCH"
    ANSWER = "ANSWER"


class AgentDefaults:
    """Значения по умолчанию для состояния агента."""
    MAX_N_ITERATIONS = 2
    N_ITERATION = 0
    USER_QUESTION = ""
    THOUGHTS = []
    CONTEXT = []
    QUESTIONS = []
    FINAL_ANSWER = ""
    STOP = False
    NEXT_STEP = NextStep.SEARCH.value
