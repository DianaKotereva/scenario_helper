# Pydantic валидация выходов агентов

## Обзор

Все агенты в системе теперь используют Pydantic модели для валидации выходных данных. Это обеспечивает:

1. **Типобезопасность** - автоматическая проверка типов данных
2. **Валидацию** - проверка корректности данных на уровне модели
3. **Документацию** - модели служат документацией структуры данных
4. **Обработку ошибок** - четкие сообщения об ошибках валидации

## Модели выходных данных

### ReasoningOutput

Выходные данные агента рассуждений (`reasoning_agent`).

```python
from src.agent.prompts.output_models import ReasoningOutput

# Пример использования
result = reasoning_agent.invoke(
    user_question="Вопрос пользователя",
    context=[{"query": "вопрос", "answer": "ответ"}]
)

# result - это экземпляр ReasoningOutput
assert isinstance(result, ReasoningOutput)
print(result.reasoning)  # str, минимум 10 символов
print(result.next_step)  # NextStep enum: SEARCH или ANSWER
print(result.to_collect)  # str, минимум 5 символов
```

**Поля:**
- `reasoning` (str): Детальные рассуждения агента (минимум 10 символов)
- `next_step` (NextStep): Решение о следующем шаге (SEARCH или ANSWER)
- `to_collect` (str): Описание информации для сбора (минимум 5 символов)

**Валидация:**
- `reasoning` проверяется на минимальную длину
- `to_collect` проверяется на соответствие `next_step` (особенно важно при SEARCH)
- Все поля автоматически очищаются от лишних пробелов

### QuestionGeneratorOutput

Выходные данные агента генерации вопросов (`questions_agent`).

```python
from src.agent.prompts.output_models import QuestionGeneratorOutput

result = questions_agent.invoke(
    user_question="Вопрос",
    reasoning="Нужно найти информацию о...",
    context=[]
)

# result - это экземпляр QuestionGeneratorOutput
assert isinstance(result, QuestionGeneratorOutput)
print(result.reasoning)  # str, минимум 10 символов
print(result.questions)  # List[str], максимум 3 вопроса, каждый минимум 5 символов
```

**Поля:**
- `reasoning` (str): Размышления агента (минимум 10 символов)
- `questions` (List[str]): Список вопросов (максимум 3, каждый минимум 5 символов)

**Валидация:**
- Проверка на дубликаты вопросов (case-insensitive)
- Автоматическое удаление пустых вопросов
- Проверка минимальной длины каждого вопроса

### RetrieveOutput

Выходные данные агента поиска (`retrieve_agent`).

```python
from src.agent.prompts.output_models import RetrieveOutput

result = retrieve_agent.invoke(query="Поисковый запрос")

# result - это экземпляр RetrieveOutput
assert isinstance(result, RetrieveOutput)
print(result.answer)  # str, минимум 20 символов
```

**Поля:**
- `answer` (str): Детальный ответ на основе найденной информации (минимум 20 символов)

**Валидация:**
- Проверка минимальной длины ответа
- Проверка на сообщения об ошибках без полезного контента

### FinalAnswerOutput

Выходные данные агента финального ответа (`answer_agent`).

```python
from src.agent.prompts.output_models import FinalAnswerOutput

result = answer_agent.invoke(
    user_question="Вопрос",
    thoughts=["Размышление 1", "Размышление 2"],
    context=[{"query": "вопрос", "answer": "ответ"}]
)

# result - это экземпляр FinalAnswerOutput
assert isinstance(result, FinalAnswerOutput)
print(result.final_answer)  # str, минимум 50 символов, минимум 2 предложения
```

**Поля:**
- `final_answer` (str): Финальный ответ (минимум 50 символов, минимум 2 предложения)

**Валидация:**
- Проверка минимальной длины
- Проверка на наличие минимум 2 предложений
- Проверка на сообщения об ошибках

## Обработка ошибок валидации

При ошибке валидации агенты используют fallback логику:

```python
try:
    result = reasoning_agent.invoke(...)
    # result - валидированный ReasoningOutput
except Exception as e:
    # Ошибка валидации обрабатывается внутри агента
    # Возвращаются значения по умолчанию
    logger.error(f"Validation error: {e}")
```

В узлах графа ошибки валидации обрабатываются автоматически:

```python
# В reasoning_node
if isinstance(result, ReasoningOutput):
    # Используем валидированные данные
    state["next_step"] = result.next_step.value
else:
    # Fallback на значения по умолчанию
    state["next_step"] = NextStep.SEARCH.value
```

## Примеры использования

### Пример 1: Прямое использование агента

```python
from src.agent.prompts import reasoning_agent, ReasoningOutput

# Вызов агента
result = reasoning_agent.invoke(
    user_question="Каковы мотивы главного героя?",
    context=[{"query": "мотивы", "answer": "Герой стремится к справедливости"}]
)

# Проверка типа результата
if isinstance(result, ReasoningOutput):
    print(f"Next step: {result.next_step}")
    print(f"Reasoning: {result.reasoning}")
    print(f"To collect: {result.to_collect}")
```

### Пример 2: Обработка ошибок валидации

```python
from src.agent.prompts import questions_agent, QuestionGeneratorOutput
from pydantic import ValidationError

try:
    result = questions_agent.invoke(
        user_question="Вопрос",
        reasoning="Нужно найти...",
        context=[]
    )
    
    if isinstance(result, QuestionGeneratorOutput):
        for question in result.questions:
            print(f"Question: {question}")
    else:
        print("Unexpected result type")
        
except ValidationError as e:
    print(f"Validation error: {e}")
    # Агент автоматически обработает ошибку и вернет fallback значения
```

### Пример 3: Использование в узлах графа

```python
# В agent_nodes.py
from src.agent.prompts.output_models import ReasoningOutput

def reasoning_node(state: AgentState) -> AgentState:
    result = reasoning_agent.invoke(...)
    
    # Проверка типа и использование валидированных данных
    if isinstance(result, ReasoningOutput):
        state["next_step"] = result.next_step.value
        state["thoughts"].append(result.reasoning)
        state["to_collect"] = result.to_collect
    else:
        # Fallback логика
        state["next_step"] = NextStep.SEARCH.value
    
    return state
```

## Преимущества Pydantic валидации

1. **Автоматическая проверка типов** - ошибки типов обнаруживаются сразу
2. **Валидация данных** - проверка корректности на уровне модели
3. **Четкие сообщения об ошибках** - понятные сообщения при валидации
4. **Документация** - модели служат документацией структуры данных
5. **IDE поддержка** - автодополнение и проверка типов в IDE
6. **Безопасность** - защита от некорректных данных

## Миграция с JsonOutputParser

Если вы использовали `JsonOutputParser` напрямую:

**Было:**
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
result = agent.invoke(...)  # dict
answer = result.get("answer", "")
```

**Стало:**
```python
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from src.agent.prompts.output_models import RetrieveOutput

parser = PydanticOutputParser(pydantic_object=RetrieveOutput)
result = agent.invoke(...)  # RetrieveOutput
answer = result.answer  # Типобезопасный доступ
```

## Тестирование

Пример теста валидации:

```python
import pytest
from pydantic import ValidationError
from src.agent.prompts.output_models import ReasoningOutput, NextStep

def test_reasoning_output_validation():
    # Валидные данные
    valid = ReasoningOutput(
        reasoning="Это валидные рассуждения длиной более 10 символов",
        next_step=NextStep.SEARCH,
        to_collect="Нужно собрать информацию о мотивах героя"
    )
    assert valid.next_step == NextStep.SEARCH
    
    # Невалидные данные - слишком короткий reasoning
    with pytest.raises(ValidationError):
        ReasoningOutput(
            reasoning="Коротко",  # Меньше 10 символов
            next_step=NextStep.SEARCH,
            to_collect="Собрать информацию"
        )
```

## Дополнительная информация

- [Pydantic документация](https://docs.pydantic.dev/)
- [LangChain PydanticOutputParser](https://python.langchain.com/docs/modules/model_io/output_parsers/structured#pydantic-output-parser)
