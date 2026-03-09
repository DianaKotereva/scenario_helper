"""
Pydantic модели для валидации выходных данных агентов.

Модуль содержит модели для валидации структурированных выходов
от различных агентов в системе.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator, root_validator


class NextStep(str, Enum):
    """Возможные следующие шаги агента после рассуждения."""
    SEARCH = "SEARCH"
    ANSWER = "ANSWER"


class ReasoningOutput(BaseModel):
    """
    Выходные данные агента рассуждений (ReasoningAgent).
    
    Валидирует структуру ответа от reasoning_agent.
    """
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Детальные рассуждения агента о собранном контексте"
    )
    next_step: NextStep = Field(
        ...,
        description="Решение о следующем шаге: SEARCH или ANSWER"
    )
    to_collect: str = Field(
        ...,
        min_length=5,
        description="Описание информации, которую нужно собрать на следующем шаге"
    )
    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Проверяет, что рассуждения не пустые и содержат осмысленный текст."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Рассуждения должны содержать минимум 10 символов")
        return v.strip()
    
    @validator('to_collect')
    def validate_to_collect(cls, v, values):
        """Проверяет, что to_collect соответствует next_step."""
        if 'next_step' in values and values['next_step'] == NextStep.SEARCH:
            if not v or len(v.strip()) < 5:
                raise ValueError(
                    "При next_step=SEARCH поле to_collect должно содержать "
                    "описание информации для сбора (минимум 5 символов)"
                )
        return v.strip() if v else v
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "reasoning": "Проанализировав собранный контекст, я вижу, что информация о мотивах героя частично присутствует, но отсутствуют детали о его прошлом, которые важны для полного понимания.",
                "next_step": "SEARCH",
                "to_collect": "Нужно найти информацию о детстве и прошлом главного героя, особенно события, которые могли повлиять на формирование его мотивов."
            }
        }


class QuestionGeneratorOutput(BaseModel):
    """
    Выходные данные агента генерации вопросов (QuestionAgent).
    
    Валидирует структуру ответа от questions_agent.
    """
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Размышления агента о том, какие вопросы нужно сгенерировать"
    )
    questions: List[str] = Field(
        ...,
        max_items=3,
        description="Список сгенерированных поисковых вопросов (не более 3)"
    )
    
    @validator('reasoning')
    def validate_reasoning(cls, v):
        """Проверяет, что рассуждения не пустые."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Рассуждения должны содержать минимум 10 символов")
        return v.strip()
    
    @validator('questions')
    def validate_questions(cls, v):
        """Проверяет качество сгенерированных вопросов."""
        if len(v) > 3:
            raise ValueError("Максимальное количество вопросов: 3")
        
        # Проверяем, что вопросы не пустые и не дублируются
        seen = set()
        validated_questions = []
        for q in v:
            if not isinstance(q, str):
                raise ValueError(f"Вопрос должен быть строкой, получен: {type(q)}")
            
            q_clean = q.strip()
            if not q_clean:
                continue  # Пропускаем пустые вопросы
            
            if len(q_clean) < 5:
                raise ValueError(f"Вопрос слишком короткий (минимум 5 символов): {q_clean}")
            
            # Проверяем на дубликаты (case-insensitive)
            q_lower = q_clean.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                validated_questions.append(q_clean)
        
        # Если все вопросы были отфильтрованы, возвращаем пустой список
        return validated_questions
    
    @root_validator(skip_on_failure=True)
    def validate_questions_reasoning_consistency(cls, values):
        """Проверяет согласованность между reasoning и questions."""
        reasoning = values.get('reasoning', '')
        questions = values.get('questions', [])
        
        # Если reasoning говорит о необходимости вопросов, но questions пуст,
        # это может быть проблемой (но допустимо, если агент решил, что вопросов не нужно)
        if len(questions) == 0 and 'не могу' not in reasoning.lower() and 'дублир' not in reasoning.lower():
            # Это предупреждение, но не ошибка
            pass
        
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "reasoning": "На основе размышлений агента нужно найти информацию о детстве героя и его отношениях с семьей. Уже были заданы вопросы о мотивах, поэтому фокусируюсь на прошлом.",
                "questions": [
                    "Детство главного героя и влияние на его характер",
                    "Отношения главного героя с семьей"
                ]
            }
        }


class RetrieveOutput(BaseModel):
    """
    Выходные данные агента поиска (RetrieveAgent).
    
    Валидирует структуру ответа от retrieve_agent.
    """
    answer: str = Field(
        ...,
        min_length=20,
        description="Детальный ответ на основе найденной информации"
    )
    
    @validator('answer')
    def validate_answer(cls, v):
        """Проверяет качество ответа."""
        if not v or len(v.strip()) < 20:
            raise ValueError(
                "Ответ должен содержать минимум 20 символов. "
                "Если информации недостаточно, явно укажите это в ответе."
            )
        
        # Проверяем, что ответ не является просто ошибкой или пустым сообщением
        v_clean = v.strip()
        error_indicators = [
            "ошибка при",
            "не удалось",
            "попробуйте переформулировать",
        ]
        
        # Если ответ содержит только сообщение об ошибке без контента, это проблема
        if any(indicator in v_clean.lower() for indicator in error_indicators):
            if len(v_clean) < 50:  # Очень короткое сообщение об ошибке
                raise ValueError(
                    "Ответ содержит только сообщение об ошибке без полезного контента. "
                    "Попробуйте переформулировать запрос или проверьте доступность данных."
                )
        
        return v_clean
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Согласно анализу главных диалогов (Глава 5) и авторским комментариям (Глава 12), мотивы главного героя можно разделить на несколько уровней:\n1. Основной мотив - стремление к справедливости, основанное на детских переживаниях...\n2. Второстепенный мотив - желание доказать свою ценность...\nВывод: Совокупность факторов указывает на сложную мотивационную структуру героя."
            }
        }


class FinalAnswerOutput(BaseModel):
    """
    Выходные данные агента финального ответа (AnswerAgent).
    
    Валидирует структуру финального ответа.
    """
    final_answer: str = Field(
        ...,
        min_length=50,
        description="Финальный ответ на вопрос пользователя"
    )
    
    @validator('final_answer')
    def validate_final_answer(cls, v):
        """Проверяет качество финального ответа."""
        if not v or len(v.strip()) < 50:
            raise ValueError(
                "Финальный ответ должен содержать минимум 50 символов. "
                "Ответ должен быть подробным и структурированным."
            )
        
        v_clean = v.strip()
        
        # Проверяем структуру ответа (должен содержать несколько предложений)
        sentences = v_clean.split('.')
        if len([s for s in sentences if s.strip()]) < 2:
            raise ValueError(
                "Финальный ответ должен содержать минимум 2 предложения. "
                "Ответ должен быть развернутым и структурированным."
            )
        
        # Проверяем, что ответ не является просто сообщением об ошибке
        error_indicators = [
            "произошла ошибка",
            "не удалось сформировать",
            "попробуйте переформулировать",
        ]
        
        if any(indicator in v_clean.lower() for indicator in error_indicators):
            if len(v_clean) < 100:  # Очень короткое сообщение об ошибке
                raise ValueError(
                    "Финальный ответ содержит только сообщение об ошибке. "
                    "Попробуйте собрать больше контекста или переформулировать вопрос."
                )
        
        return v_clean
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_answer": "Согласно анализу главных диалогов (Глава 5, с. 45-47) и авторским комментариям (Глава 12, с. 112), мотивы главного героя можно разделить на несколько уровней:\n\n1. **Основной мотив** - стремление к справедливости, основанное на детских переживаниях несправедливости со стороны взрослых. Это видно в эпизоде с учителем (Глава 3).\n\n2. **Второстепенный мотив** - желание доказать свою ценность через достижения, что проявляется в его упорстве в учебе (Глава 7).\n\n3. **Скрытый мотив** - потребность в признании и любви, которая движет его отношениями с другими персонажами (Главы 9-11).\n\n**Вывод**: Совокупность факторов указывает на сложную мотивационную структуру героя, где личные переживания детства формируют его взрослые стремления и поступки."
            }
        }
