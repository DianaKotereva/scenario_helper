import json
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers import JsonOutputParser

from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookNode


system_prompt = """
Ты — агент верификации сущностей в литературном графе знаний.
Нужно определить, относятся ли два описания к одной и той же сущности.

Важно:
1) Совпадение имени само по себе недостаточно.
2) Для тёзок разных поколений (например, «отец/сын/внук») по умолчанию ставь `is_same_entity=false`.
3) Разрешай merge тёзок только при явном текстовом доказательстве одной идентичности:
- прямое указание на переименование/алиас,
- явное подтверждение, что это тот же персонаж в иной роли/форме.
4) Если есть конфликт родственных якорей, ставь `is_same_entity=false` и заполняй `hard_conflict_flags`.

Критерии merge (должны быть совместимы):
- классификация сущности;
- уникальные идентификаторы (main_name/устойчивые alias);
- совместимость действий и биографии;
- отсутствие противоречий по родству/поколению/роли.

Автоматический reject:
- разные classification;
- конфликт уникальных идентификаторов без явного моста;
- конфликт поколений/родства;
- взаимно исключающие биографические факты.

Требуемый JSON-ответ (строго, без дополнительного текста):
{
  "is_same_entity": bool,
  "confidence": "высокий|средний|низкий",
  "key_evidence": [str, ...],
  "conflicting_attributes": [str, ...],
  "hard_conflict_flags": [str, ...],
  "kinship_anchors": {
    "child_of": [str, ...],
    "parent_of": [str, ...],
    "grandchild_of": [str, ...]
  },
  "merge_blocked_by": str | null
}

Правила для `kinship_anchors`:
- заполняй только по данным из входа (без выдумывания),
- если данных нет, возвращай пустые списки.
"""


class Verification(LLMBase):
    """Класс для верификации сущностей."""

    def __init__(self, llm, parser: JsonOutputParser = None):
        if parser is None:
            parser = JsonOutputParser()
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            parser=parser,
            parse_json=True,
        )

    def make_user_prompt(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int, ...],
        last_n: int = -10,
        existing_kinship_aliases: List[str] | None = None,
        new_kinship_aliases: List[str] | None = None,
    ) -> Dict[str, Any]:
        main_name = node.main_name
        alt_names = node.alt_names
        classification = node.classification
        all_prev_actions = ". ".join([i.action for i in node.actions][last_n:])

        payload = {
            "task": "entity_verification",
            "source_id": list(source_id),
            "existing_entity": {
                "main_name": main_name,
                "alt_names": alt_names,
                "kinship_aliases": existing_kinship_aliases or [],
                "classification": classification,
                "actions": all_prev_actions,
            },
            "new_entity": {
                "main_name": new_node.get("main_name", ""),
                "alt_names": new_node.get("alt_names", []),
                "kinship_aliases": new_kinship_aliases or [],
                "classification": new_node.get("classification", ""),
                "actions": new_node.get("actions", ""),
            },
            "output_schema_hint": {
                "is_same_entity": "bool",
                "confidence": "высокий|средний|низкий",
                "key_evidence": "list[str]",
                "conflicting_attributes": "list[str]",
                "hard_conflict_flags": "list[str]",
                "kinship_anchors": {
                    "child_of": "list[str]",
                    "parent_of": "list[str]",
                    "grandchild_of": "list[str]",
                },
                "merge_blocked_by": "str|null",
            },
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        return {"messages": [("user", user_prompt)]}
