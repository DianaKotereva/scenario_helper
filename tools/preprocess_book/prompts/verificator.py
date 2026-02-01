from typing import Dict, Any, Tuple
from langchain_core.output_parsers import JsonOutputParser
from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookNode


system_prompt = """
**Ты - Агент верификации сущностей. Твоя задача - определить, относятся ли существующее и новое описания к ОДНОЙ И ТОЙ ЖЕ сущности (персонаж/место/объект).**
Учти, что результат верификации будет использован в создании Голливудских фильмов, поэтому если ты плохо проверишь, то тебя подвергнут большому штрафу и суду.

**Входные данные:**
1. `Существующее описание` (из базы знаний): 
   - main_name, classification, actions, контекстные атрибуты
2. `Новое описание` (из текущей главы): 
   - main_name, classification, actions, контекстные атрибуты

**Правила анализа:**
1. **Критерии идентичности** (все условия ДОЛЖНЫ выполняться):
   - ✅ Совпадение `main_name` ИЛИ явное указание на переименование
   - ✅ Совпадение `classification` (персонаж/место/организация и т.д.)
   - ✅ Семантическая согласованность `actions` (новые действия логично продолжают старые)
   - ✅ Отсутствие **конфликтующих атрибутов** (противоречивые описания внешности, места, статуса)

2. **Автоматический отказ** (если ЛЮБОЕ из условий):
   - ❌ Разные `classification` (например: существующее - "персонаж", новое - "место")
   - ❌ Конфликт уникальных идентификаторов (разные имена без указания переименования)
   - ❌ Несовместимые физические/временные параметры (персонаж в двух местах одновременно без объяснения)

3. **Контекстные нюансы:**
   - Учитывай альтернативные имена (`alt_names`) при проверке
   - Разрешай правдоподобное развитие атрибутов (изменение внешности, статуса)
   - Игнорируй общеупотребительные эпитеты ("красивый", "старый") как нерелевантные

**Выходной формат (СТРОГО JSON):**
```json
{{
  "is_same_entity": bool,
  "confidence": "высокий/средний/низкий",
  "key_evidence": [список строк с ключевыми подтверждающими фактами],
  "conflicting_attributes": [список строк с несовместимыми атрибутами]
}}
```

### Примеры решений:
1️⃣ **Идентичные сущности**  
Существующее: {{"main_name": "Замок Дракона", "classification": "место", "actions": "Логово дракона Нидхёгга"}}  
Новое: {{"main_name": "Драконье Гнездо", "classification": "место", "actions": "Обитель Нидхёгга"}}  
➔ Выход: {{"is_same_entity": true, "confidence": "высокий", ...}}

2️⃣ **Разные сущности**  
Существующее: {{"main_name": "Капитан Рейв", "classification": "персонаж", "actions": "Добрый паладин и страж закона"}}  
Новое: {{"main_name": "Капитан Рейв", "classification": "персонаж", "actions": "Возглавляет гильдию воров"}}  
➔ Выход: {{"is_same_entity": false, "conflicting_attributes": ["Добрый паладин vs Возглавляет гильдию"]}}

3️⃣ **Пограничный случай**  
Существующее: {{"main_name": "Аэлита", "alt_names": ["Королева Теней"], ...}}  
Новое: {{"main_name": "Неизвестная ассасин", ...}}  
➔ Выход: {{"is_same_entity": false, "confidence": "низкий", "key_evidence": ["Нет совпадающих уникальных идентификаторов"]}}"""


class Verification(LLMBase):
    """Класс для верификации сущностей."""
    
    def __init__(self, llm, parser: JsonOutputParser = None):
        """
        Args:
            llm: Инициализированная языковая модель
            parser: Парсер для JSON вывода (по умолчанию JsonOutputParser)
        """
        if parser is None:
            parser = JsonOutputParser()
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            parser=parser,
            parse_json=True
        )
    
    def make_user_prompt(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int],
        last_n: int = -10
    ) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для верификации.
        
        Args:
            node: Существующий узел BookNode
            new_node: Новый узел в виде словаря
            source_id: ID источника
            last_n: Количество последних действий для сравнения
            
        Returns:
            Словарь с сообщениями для LLM
        """
        main_name = node.main_name
        alt_names = node.alt_names
        classification = node.classification
        all_prev_actions = ". ".join([i.action for i in node.actions][last_n:])
        
        user_prompt = [
            "**Существующее**: {{",
            f'"main_name": {main_name},',
            f'"alt_names": {alt_names}',
            f'"classification": {classification}',
            f'"actions": {all_prev_actions}',
            "}}",
            "\n\n\n",
            "**Новое**: {{",
            f'"main_name": {new_node["main_name"]}, ',
            f'"alt_names": {new_node["alt_names"]}',
            f'"classification": {new_node["classification"]}',
            f'"actions": {new_node["actions"]}',
            "}}",
        ]
        user_prompt = "\n".join(user_prompt)
        return {"messages": [("user", user_prompt)]}
