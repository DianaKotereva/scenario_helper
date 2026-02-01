"""
Промпт и агент для генерации поисковых вопросов.

Модуль содержит промпт и класс QuestionAgent для генерации
уточняющих вопросов на основе размышлений агента и уже собранного контекста.
"""

import logging
from typing import Dict, Any, List

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.agent.prompts.output_models import QuestionGeneratorOutput

logger = logging.getLogger(__name__)

question_generator_prompt = """Ты — профессиональный генератор поисковых запросов для RAG системы по книге. 
Ты должен генерировать вопросы по книге.

# Задача:
Твои входные данные:
1. Исходный вопрос пользователя: str // Исходный вопрос, который задал пользователь
2. Размышления агента: str // Инструкция, по которой ты должен сгенерировать вопросы.
3. Вопросы, на которые уже был дан ответ: List[Dict[str, str]] // История ответов, которые уже были даны

# Требования к вопросам:
0. Вопросы должны дать максимально полные ответы на вопросы пользователя
1. Использование терминов из контекста размышлений
2. Формат как для поисковой системы (без предложений-ответов)
3. Каждый вопрос должен уточнять разные аспекты
4. Избегай общих вопросов в стиле "расскажи всё о..."
5. Вопрос должен быть К СЮЖЕТУ книги, к персонажам. Тебе запрещено использовать вопросы формата "Роль Сони в сюжете".
6. Учти вопросы, которые уже были заданы! Твои вопросы не должны дублировать те, что уже были созданы! Если ты не можешь придумать новых вопросов, в поле questions верни пустой список.
7. Используй точные термины из контекста при генерации вопросов.

# Формат вывода:
Ты должен вернуть JSON со следующими полями: 

{
  "reasoning": str, // Твои размышления (минимум 10 символов)
  "questions": list[str] // Не более 3 вопросов. Каждый вопрос минимум 5 символов. Если не можешь придумать новых вопросов, верни пустой список.
}

# ТРЕБОВАНИЯ К ВАЛИДАЦИИ:
- reasoning: минимум 10 символов
- questions: максимум 3 вопроса, каждый минимум 5 символов
- Вопросы не должны дублироваться (case-insensitive)
- Если все вопросы дублируются, верни пустой список

{{format_instructions}}

Сгенерируй вопросы:"""


class QuestionAgent(LLMBase):
    """
    Агент для генерации поисковых вопросов.
    
    Генерирует уточняющие вопросы на основе размышлений агента
    и истории уже заданных вопросов, чтобы избежать дублирования.
    """
    
    def make_user_prompt(
        self, 
        user_question: str, 
        reasoning: str, 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для генерации вопросов.
        
        Args:
            user_question: Исходный вопрос пользователя
            reasoning: Размышления агента о том, какую информацию нужно собрать
            context: История уже заданных вопросов и полученных ответов
            
        Returns:
            Словарь с сообщениями для LLM в формате {"messages": [("user", prompt)]}
            
        Raises:
            ValueError: Если входные данные некорректны
        """
        if not user_question or not isinstance(user_question, str):
            raise ValueError("user_question must be a non-empty string")
        
        if not isinstance(reasoning, str):
            reasoning = ""
        
        if not isinstance(context, list):
            context = []
        
        message = [
            f"**Исходный вопрос пользователя:** {user_question}",
            f"**Размышления агента:** {reasoning}",
            f"**Уже отвеченные вопросы:** {context}",
        ]
        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}
        
        logger.debug(f"Formatted prompt for question generation (context items: {len(context)})")
        return messages


# Создаем парсер с Pydantic моделью для валидации
questions_parser = PydanticOutputParser(pydantic_object=QuestionGeneratorOutput)

# Обновляем промпт с инструкциями по форматированию
question_generator_prompt_with_format = question_generator_prompt.replace(
    "{{format_instructions}}",
    questions_parser.get_format_instructions()
)

# Создаем экземпляр агента с валидацией
questions_agent = QuestionAgent(llm, question_generator_prompt_with_format, questions_parser)
