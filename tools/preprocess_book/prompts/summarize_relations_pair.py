from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.llm_core.llm_prompt_base import LLMBase

system_prompt = """
Ты аналитик литературного графа.

Твоя задача: по двум сущностям и списку всех наблюдений об их связях
сформировать краткую и точную суммаризацию отношений.

Вход:
- source_entity: {main_name, alt_names, profile_summary}
- target_entity: {main_name, alt_names, profile_summary}
- relations: список объектов:
  {type, description, source_id, chapter_id}

Сделай:
1) relation_summary: 2-5 предложений, что между ними происходит по сюжету.
2) kinship_summary: если есть явные родственные связи (отец/сын/мать/дочь/брат/сестра/супруг),
   опиши их. Если данных нет — "не указано".
3) relation_types: уникальный список ключевых типов отношений (до 10).
4) confidence: high/medium/low по полноте и однозначности.

Правила:
- Не выдумывай фактов, опирайся только на вход.
- profile_summary используй как вспомогательный контекст для дизамбигуации имён.
- Пиши на русском.
- Верни только JSON.
"""


class RelationEvidenceItem(BaseModel):
    type: str = Field(default="")
    description: str = Field(default="")
    source_id: int | None = Field(default=None)
    chapter_id: int | None = Field(default=None)


class EntityRef(BaseModel):
    main_name: str = Field(default="")
    alt_names: List[str] = Field(default_factory=list)
    profile_summary: str = Field(default="")


class RelationPairSummary(BaseModel):
    relation_summary: str = Field(default="")
    kinship_summary: str = Field(default="не указано")
    relation_types: List[str] = Field(default_factory=list)
    confidence: str = Field(default="medium")


class SummarizeRelationsPair(LLMBase):
    def __init__(self, llm, parser: PydanticOutputParser | None = None):
        if parser is None:
            parser = PydanticOutputParser(pydantic_object=RelationPairSummary)
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            parser=parser,
            parse_json=True,
        )

    def make_user_prompt(
        self,
        source_entity: Dict[str, Any],
        target_entity: Dict[str, Any],
        relations: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "source_entity": EntityRef(**(source_entity or {})).model_dump(mode="json"),
            "target_entity": EntityRef(**(target_entity or {})).model_dump(mode="json"),
            "relations": [RelationEvidenceItem(**(r or {})).model_dump(mode="json") for r in relations],
        }
        return {"messages": [("user", json.dumps(payload, ensure_ascii=False))]}
