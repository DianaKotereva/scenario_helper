"""
Промпт и агент для рассуждений и принятия решений.

Модуль содержит промпт и класс ReasoningAgent для анализа
собранного контекста и принятия решения о следующем шаге.
"""

import logging
from typing import Dict, Any, List

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.agent.prompts.output_models import ReasoningOutput

logger = logging.getLogger(__name__)

reasoning_prompt = """Ты — аналитический агент-рассуждатель. Твоя задача — критически оценить собранную информацию и принять решение о дальнейших действиях.

## Твои входные данные: 
**1. Исходный вопрос пользователя**: str //Исходный вопрос, который задал пользователь
**2. Собранный контекст**: List[str] // Уже имеющийся у тебя контекст

# Инструкции:
1. Проанализируй контекст на соответствие исходному вопросу
2. Оцени полноту информации по критериям:
   - Покрыты ли все аспекты вопроса?
   - Есть ли противоречия в данных?
   - Присутствуют ли неясные моменты?
   - Какой информации не хватает, чтобы дать полный, четкий и ясный ответ на вопрос?
3. Прими решение по следующей логике:
   * Если контекст НЕ ПОЛНОСТЬЮ отвечает на вопрос -> SEARCH
   * Если контекст ДОСТАТОЧЕН и НЕТ ПРОТИВОРЕЧИЙ -> ANSWER
4. Если ты понимаешь, что для исчерпывающего ответа на вопрос требуется больше информации, напиши, какая дополнительная информация тебе нужна. 
   - Опирайся на контекст, задавай вопрос и делай описание исходя из той контекстной информации, которая уже есть
   - Постарайся давать конкретные задачи по сбору информации, избегай слишком расплывчатых задач.

# Выходные данные: 
Ты должен вернуть JSON со следующими полями: 

{
  "reasoning": str, // Твои детальные рассуждения (минимум 10 символов)
  "next_step": "SEARCH" | "ANSWER", // Если требуется собрать еще информацию -> SEARCH, если достаточно -> ANSWER
  "to_collect": str // Описание задания для сбора информации (минимум 5 символов, особенно важно при next_step=SEARCH)
}

# ТРЕБОВАНИЯ К ВАЛИДАЦИИ:
- reasoning: минимум 10 символов, должен содержать осмысленный анализ
- next_step: строго "SEARCH" или "ANSWER"
- to_collect: минимум 5 символов, особенно важно при next_step=SEARCH

{{format_instructions}}

"""


class ReasoningAgent(LLMBase):
    """
    Агент для рассуждений и принятия решений.
    
    Анализирует собранный контекст и определяет, достаточно ли информации
    для формирования ответа или требуется дополнительный поиск.
    """
    
    def make_user_prompt(
        self, 
        user_question: str, 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для рассуждений.
        
        Args:
            user_question: Исходный вопрос пользователя
            context: Собранный контекст (список словарей с вопросами и ответами)
            
        Returns:
            Словарь с сообщениями для LLM в формате {"messages": [("user", prompt)]}
            
        Raises:
            ValueError: Если входные данные некорректны
        """
        if not user_question or not isinstance(user_question, str):
            raise ValueError("user_question must be a non-empty string")
        
        if not isinstance(context, list):
            context = []
        
        message = [
            f"**Исходный вопрос пользователя:** {user_question}",
            f"**Собранный контекст:** {context}",
        ]
        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}
        
        logger.debug(f"Formatted prompt for reasoning (context items: {len(context)})")
        return messages


# Создаем парсер с Pydantic моделью для валидации
reasoning_parser = PydanticOutputParser(pydantic_object=ReasoningOutput)

# Обновляем промпт с инструкциями по форматированию
reasoning_prompt_with_format = reasoning_prompt.replace(
    "{{format_instructions}}",
    reasoning_parser.get_format_instructions()
)

# Создаем экземпляр агента с валидацией
reasoning_agent = ReasoningAgent(llm, reasoning_prompt_with_format, reasoning_parser)
