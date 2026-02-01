"""
Узлы графа агента для обработки вопросов пользователя.

Модуль содержит функции-узлы для LangGraph workflow:
- reasoning_node: Анализ контекста и определение следующего шага
- search_node: Поиск информации по сгенерированным вопросам
- final_answer_node: Формирование финального ответа
- cond_edge_reasoner: Условное ветвление на основе решения reasoning_node
"""

import logging
from copy import deepcopy

import src.config.settings as settings
from src.agent.agent_graph.constants import AgentDefaults, NextStep
from src.agent.agent_graph.states.agent_state import AgentState
from src.agent.prompts import (
    answer_agent,
    questions_agent,
    reasoning_agent,
    retrieve_agent,
)

logger = logging.getLogger(__name__)

# Значения по умолчанию для состояния агента
default_values = [
    ("max_n_iterations", settings.MAX_N_ITERATIONS),
    ("n_iteration", AgentDefaults.N_ITERATION),
    ("user_question", AgentDefaults.USER_QUESTION),
    ("thoughts", AgentDefaults.THOUGHTS),
    ("context", AgentDefaults.CONTEXT),
    ("questions", AgentDefaults.QUESTIONS),
    ("final_answer", AgentDefaults.FINAL_ANSWER),
    ("stop", AgentDefaults.STOP),
]


def _initialize_default_values(state: AgentState) -> AgentState:
    """
    Инициализирует значения по умолчанию для состояния агента.

    Args:
        state: Текущее состояние агента

    Returns:
        Состояние с инициализированными значениями по умолчанию
    """
    for key, value in deepcopy(default_values):
        if key not in state:
            state[key] = deepcopy(value) if isinstance(value, list) else value
    return state


def _validate_state(state: AgentState) -> None:
    """
    Валидирует состояние агента.

    Args:
        state: Состояние для валидации

    Raises:
        ValueError: Если состояние некорректно
    """
    if not state.get("user_question"):
        raise ValueError("user_question is required in state")

    if "n_iteration" in state and state["n_iteration"] < 0:
        raise ValueError("n_iteration must be non-negative")

    if "max_n_iterations" in state and state["max_n_iterations"] < 1:
        raise ValueError("max_n_iterations must be at least 1")


def reasoning_node(state: AgentState) -> AgentState:
    """
    Узел рассуждения, анализирующий контекст и определяющий следующий шаг.

    Анализирует собранный контекст и принимает решение о необходимости
    дополнительного поиска информации или формировании финального ответа.

    Args:
        state: Текущее состояние агента

    Returns:
        Обновленное состояние агента с решением о следующем шаге

    Raises:
        ValueError: Если состояние некорректно
        RuntimeError: Если вызов LLM не удался
    """
    try:
        # Инициализация значений по умолчанию
        state = _initialize_default_values(state)

        # Валидация состояния
        _validate_state(state)

        # Если есть контекст, анализируем его
        if state.get("context"):
            try:
                result = reasoning_agent.invoke(
                    user_question=state["user_question"],
                    context=state.get("context", []),
                )

                # Обработка результата с Pydantic валидацией
                from src.agent.prompts.output_models import ReasoningOutput

                # Если результат уже является Pydantic моделью (после парсинга)
                if isinstance(result, ReasoningOutput):
                    reasoning_output = result
                elif isinstance(result, dict):
                    # Пытаемся создать модель из словаря (валидация)
                    try:
                        reasoning_output = ReasoningOutput(**result)
                    except Exception as validation_error:
                        logger.error(
                            f"Validation error in reasoning output: {validation_error}",
                            exc_info=True,
                        )
                        # Fallback на значения по умолчанию
                        state["next_step"] = NextStep.SEARCH.value
                        state["to_collect"] = "Ошибка валидации результата рассуждений"
                        state["n_iteration"] = state.get("n_iteration", 0) + 1
                        return state
                else:
                    logger.warning(
                        f"Reasoning agent returned unexpected type: {type(result)}"
                    )
                    state["next_step"] = NextStep.SEARCH.value
                    state["n_iteration"] = state.get("n_iteration", 0) + 1
                    return state

                # Используем валидированные значения из Pydantic модели
                state["thoughts"].append(reasoning_output.reasoning)
                state["to_collect"] = reasoning_output.to_collect
                state["next_step"] = reasoning_output.next_step.value  # Enum -> str

                logger.debug(
                    f"Reasoning result validated: next_step={reasoning_output.next_step}, "
                    f"reasoning_length={len(reasoning_output.reasoning)}"
                )

            except Exception as e:
                logger.error(f"Error in reasoning_agent.invoke: {e}", exc_info=True)
                state["next_step"] = NextStep.SEARCH.value
                state["to_collect"] = "Ошибка при анализе контекста"
        else:
            # Если контекста нет, переходим к поиску
            state["next_step"] = NextStep.SEARCH.value

        # Увеличиваем счетчик итераций
        state["n_iteration"] = state.get("n_iteration", 0) + 1

        return state

    except ValueError as e:
        logger.error(f"Validation error in reasoning_node: {e}", exc_info=True)
        state["next_step"] = NextStep.ANSWER.value
        state["stop"] = True
        return state
    except Exception as e:
        logger.error(f"Unexpected error in reasoning_node: {e}", exc_info=True)
        state["next_step"] = NextStep.ANSWER.value
        state["stop"] = True
        return state


def cond_edge_reasoner(state: AgentState) -> str:
    """
    Условное ветвление на основе решения reasoning_node.

    Определяет, нужно ли продолжить поиск информации или перейти
    к формированию финального ответа.

    Args:
        state: Текущее состояние агента

    Returns:
        Имя следующего узла: "search_node" или "final_answer_node"
    """
    try:
        next_step = state.get("next_step", NextStep.SEARCH.value)
        n_iteration = state.get("n_iteration", 0)
        max_iterations = state.get("max_n_iterations", settings.MAX_N_ITERATIONS)
        stop = state.get("stop", False)

        # Проверяем условия для продолжения поиска
        should_search = (
            next_step == NextStep.SEARCH.value
            and n_iteration <= max_iterations
            and not stop
        )

        if should_search:
            logger.debug(
                f"Continuing search (iteration {n_iteration}/{max_iterations})"
            )
            return "search_node"
        else:
            logger.debug("Moving to final answer")
            return "final_answer_node"

    except Exception as e:
        logger.error(f"Error in cond_edge_reasoner: {e}", exc_info=True)
        return "final_answer_node"


def search_node(state: AgentState) -> AgentState:
    """
    Узел поиска информации по сгенерированным вопросам.

    Генерирует вопросы для поиска (если нужно) и выполняет поиск
    информации с помощью retrieve_agent.

    Args:
        state: Текущее состояние агента

    Returns:
        Обновленное состояние агента с найденной информацией

    Raises:
        RuntimeError: Если поиск не удался
    """
    try:
        # Валидация состояния
        _validate_state(state)

        # Определяем уже выполненные запросы
        context = state.get("context", [])
        done_queries = [
            list(item.keys())[0] for item in context if isinstance(item, dict) and item
        ]

        # Генерируем вопросы для поиска
        user_question = state["user_question"]

        if user_question not in done_queries:
            questions = [user_question]
            logger.debug(f"Using original question: {user_question}")
        else:
            try:
                result = questions_agent.invoke(
                    user_question=user_question,
                    reasoning=state.get("to_collect", ""),
                    context=context,
                )

                # Обработка результата с Pydantic валидацией
                from src.agent.prompts.output_models import QuestionGeneratorOutput

                # Если результат уже является Pydantic моделью
                if isinstance(result, QuestionGeneratorOutput):
                    questions_output = result
                elif isinstance(result, dict):
                    # Пытаемся создать модель из словаря (валидация)
                    try:
                        questions_output = QuestionGeneratorOutput(**result)
                    except Exception as validation_error:
                        logger.error(
                            f"Validation error in questions output: {validation_error}",
                            exc_info=True,
                        )
                        questions = []
                    else:
                        questions = questions_output.questions
                else:
                    logger.warning(
                        f"Questions agent returned unexpected type: {type(result)}"
                    )
                    questions = []

                logger.debug(f"Generated {len(questions)} validated questions")

            except Exception as e:
                logger.error(f"Error generating questions: {e}", exc_info=True)
                questions = []

        # Выполняем поиск по каждому вопросу
        if questions:
            for query in questions:
                if not query or not isinstance(query, str):
                    logger.warning(f"Invalid query: {query}, skipping")
                    continue

                logger.info(f"Searching for: {query}")

                try:
                    result = retrieve_agent.invoke(query=query)

                    # Обработка результата с Pydantic валидацией
                    from src.agent.prompts.output_models import RetrieveOutput

                    # Если результат уже является Pydantic моделью
                    if isinstance(result, RetrieveOutput):
                        retrieve_output = result
                        answer = retrieve_output.answer
                    elif isinstance(result, dict):
                        # Пытаемся создать модель из словаря (валидация)
                        try:
                            retrieve_output = RetrieveOutput(**result)
                            answer = retrieve_output.answer
                        except Exception as validation_error:
                            logger.error(
                                f"Validation error in retrieve output for query '{query}': {validation_error}",
                                exc_info=True,
                            )
                            # Fallback: используем значение из словаря, если есть
                            answer = result.get("answer", "")
                            if not answer or len(answer) < 20:
                                logger.warning(
                                    f"Invalid answer for query '{query}', skipping"
                                )
                                continue
                    else:
                        logger.warning(
                            f"Retrieve agent returned unexpected type: {type(result)} for query: {query}"
                        )
                        continue

                    if answer:
                        state["context"].append({query: answer})
                        logger.debug(
                            f"Found validated answer for query: {query[:50]}... (length: {len(answer)})"
                        )
                    else:
                        logger.warning(f"No answer found for query: {query}")

                except Exception as e:
                    logger.error(
                        f"Error retrieving answer for query '{query}': {e}",
                        exc_info=True,
                    )
                    continue
        else:
            logger.warning("No questions to search for, stopping")
            state["stop"] = True

        return state

    except ValueError as e:
        logger.error(f"Validation error in search_node: {e}", exc_info=True)
        state["stop"] = True
        return state
    except Exception as e:
        logger.error(f"Unexpected error in search_node: {e}", exc_info=True)
        state["stop"] = True
        return state


def final_answer_node(state: AgentState) -> AgentState:
    """
    Узел формирования финального ответа на вопрос пользователя.

    Агрегирует всю собранную информацию и формирует финальный ответ
    с помощью answer_agent.

    Args:
        state: Текущее состояние агента

    Returns:
        Обновленное состояние агента с финальным ответом

    Raises:
        RuntimeError: Если формирование ответа не удалось
    """
    try:
        # Валидация состояния
        _validate_state(state)

        try:
            result = answer_agent.invoke(
                user_question=state["user_question"],
                thoughts=state.get("thoughts", []),
                context=state.get("context", []),
            )

            # Обработка результата с Pydantic валидацией
            from src.agent.prompts.output_models import FinalAnswerOutput

            # Если результат уже является Pydantic моделью
            if isinstance(result, FinalAnswerOutput):
                answer_output = result
                final_answer = answer_output.final_answer
            elif isinstance(result, dict):
                # Пытаемся создать модель из словаря (валидация)
                try:
                    answer_output = FinalAnswerOutput(**result)
                    final_answer = answer_output.final_answer
                except Exception as validation_error:
                    logger.error(
                        f"Validation error in final answer output: {validation_error}",
                        exc_info=True,
                    )
                    # Fallback: используем значение из словаря, если есть
                    final_answer = result.get("final_answer", "")
                    if not final_answer or len(final_answer) < 50:
                        logger.warning("Answer agent returned invalid final_answer")
                        final_answer = "Не удалось сформировать валидный ответ на основе доступной информации. Попробуйте переформулировать вопрос."
            else:
                logger.warning(f"Answer agent returned unexpected type: {type(result)}")
                final_answer = "Ошибка при формировании ответа."

            if final_answer and len(final_answer) >= 50:
                state["final_answer"] = final_answer
                logger.info(
                    f"Final answer generated successfully (length: {len(final_answer)})"
                )
            else:
                logger.warning(
                    f"Final answer too short or empty: {len(final_answer) if final_answer else 0}"
                )
                state["final_answer"] = (
                    "Не удалось сформировать полный ответ на основе доступной информации."
                )

        except Exception as e:
            logger.error(f"Error in answer_agent.invoke: {e}", exc_info=True)
            state["final_answer"] = (
                f"Произошла ошибка при формировании ответа: {str(e)}. "
                "Попробуйте переформулировать вопрос."
            )

        return state

    except ValueError as e:
        logger.error(f"Validation error in final_answer_node: {e}", exc_info=True)
        state["final_answer"] = "Ошибка валидации состояния агента."
        return state
    except Exception as e:
        logger.error(f"Unexpected error in final_answer_node: {e}", exc_info=True)
        state["final_answer"] = "Произошла неожиданная ошибка при формировании ответа."
        return state
