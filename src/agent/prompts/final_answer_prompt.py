"""
Промпт и агент для формирования финального ответа.

Модуль содержит промпт и класс AnswerAgent для генерации
финального ответа на основе собранного контекста и размышлений.
"""

import logging
from typing import Dict, Any, List

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.agent.prompts.output_models import FinalAnswerOutput

logger = logging.getLogger(__name__)

final_answer_prompt = """Ты — эксперт-аналитик, формирующий ответ на вопрос пользователя по книге. 

# ЗАДАЧА:
Дать ответ ИСКЛЮЧИТЕЛЬНО на основе:
1. Исходный вопрос пользователя: str //Исходный вопрос, который задал пользователь
2. Цепочка размышлений: List[str] // 
3. Собранный контекст: List[str] // Уже имеющийся у тебя контекст

# ИНСТРУКЦИИ:
1. Анализ:
   - Сопоставь каждый элемент контекста с запросом
   - Выяви ключевые доказательные факты
   - Опирайся только на ту информацию в контексте, которая имеет отношение к исходному запросу пользователя, не используй нерелевантную информацию

2. Формирование ответа:
   - Структурируй ответ: тезис → доказательства → вывод
   - Используй ТОЧНЫЕ цитаты из контекста
   - Сохрани цепочку логических умозаключений

3. Контроль качества:
   - Если информации НЕДОСТАТОЧНО → явно укажи это
   - Запрещены домыслы/интерпретации
   - Ответ должен покрывать ВСЕ аспекты запроса, быть максимально подробным, структурированным и точным. Постарайся ничего не упустить. 

# ПРИМЕРЫ:
Запрос: "Каковы мотивы главного героя в романе X?"
Правильный ответ: {{
  "final_answer": "Согласно анализу главных диалогов (с. 45-47) и авторским комментариям (с. 112):\n1. Основной мотив - ...\n2. Второстепенный мотив - ...\nВывод: Совокупность факторов указывает на..." 
}}

Недопустимый ответ: "Герой хотел добиться успеха, возможно из-за детских травм" 

# ТРЕБОВАНИЯ К ВЫВОДУ:
- ТОЛЬКО JSON-объект с ключом "final_answer"
- Ответ на языке оригинала запроса
- Четкая структура с указанием источников
- Минимум 3 доказательных пункта при наличии данных
- Объем: 150-300 слов

###
Ты должен вернуть ответ в следующем формате JSON:

{
  "final_answer": str // Финальный ответ (минимум 50 символов, минимум 2 предложения)
}

# ТРЕБОВАНИЯ К ВАЛИДАЦИИ:
- final_answer: минимум 50 символов
- Ответ должен содержать минимум 2 предложения
- Ответ должен быть структурированным: тезис → доказательства → вывод
- Если информации недостаточно, явно укажи это

{{format_instructions}}

Сформируй ответ:"""


class AnswerAgent(LLMBase):
    """
    Агент для формирования финального ответа на вопрос пользователя.
    
    Агрегирует собранный контекст и цепочку размышлений для формирования
    структурированного и обоснованного ответа.
    """
    
    def make_user_prompt(
        self, 
        user_question: str, 
        thoughts: List[str], 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для генерации финального ответа.
        
        Args:
            user_question: Исходный вопрос пользователя
            thoughts: Цепочка размышлений агента
            context: Собранный контекст (список словарей с вопросами и ответами)
            
        Returns:
            Словарь с сообщениями для LLM в формате {"messages": [("user", prompt)]}
            
        Raises:
            ValueError: Если входные данные некорректны
        """
        if not user_question or not isinstance(user_question, str):
            raise ValueError("user_question must be a non-empty string")
        
        if not isinstance(thoughts, list):
            thoughts = []
        
        if not isinstance(context, list):
            context = []
        
        # Форматируем контекст для промпта
        context_str = str(context) if context else "Контекст отсутствует"
        thoughts_str = "\n".join(thoughts) if thoughts else "Размышления отсутствуют"
        
        message = [
            f"**Исходный вопрос пользователя:** {user_question}",
            f"**Цепочка размышлений:** {thoughts_str}",
            f"**Собранный контекст:** {context_str}",
        ]
        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}
        
        logger.debug(f"Formatted prompt for final answer (question length: {len(user_question)})")
        return messages


# Создаем парсер с Pydantic моделью для валидации
answer_parser = PydanticOutputParser(pydantic_object=FinalAnswerOutput)

# Обновляем промпт с инструкциями по форматированию
final_answer_prompt_with_format = final_answer_prompt.replace(
    "{{format_instructions}}",
    answer_parser.get_format_instructions()
)

# Создаем экземпляр агента с валидацией
answer_agent = AnswerAgent(llm, final_answer_prompt_with_format, answer_parser)
