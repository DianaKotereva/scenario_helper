from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from src.llm_core.llm_prompt_base import LLMBase

CLASS_CHARACTER = "персонаж"
CLASS_PLACE = "место"
CLASS_ORG = "организация"
CLASS_TERM = "термин"
CLASS_FORCE = "сила природы"


system_prompt = f"""Ты профессиональный анализатор текстов, преобразующий художественные произведения в детализированные семантические графы.
Старайся извлекать как можно больше информации из текста, не жертвуя точностью. Не добавляй данные, которые явно не упомянуты в тексте.
Цель - обеспечить четкость и структурированность графа знаний, чтобы он в полной мере отражал содержание художественного произведения.

### Входные данные:
Текст - отрывок главы книги или несколько глав с разметкой [CHAPTER source_id=... chapter_id=...].

### Правила обработки:

Ты должен отвечать СТРОГО на русском языке.

1. Извлеки именованные сущности как узлы графа:
   - main_name: каноническое имя (выбирай наиболее устойчивое в контексте всей книги)
   - alt_names: все варианты именования (только осмысленные синонимы)
   - actions: суммарное описание действий/роли в виде СПИСКА объектов:
     {{"source_id": int, "chapter_id": int, "description": str}}
   - classification: строго один из ['{CLASS_CHARACTER}', '{CLASS_PLACE}', '{CLASS_ORG}', '{CLASS_TERM}', '{CLASS_FORCE}']

2. Сделай небольшую суммаризацию текста не более 1-2 абзацев.

3. Приоритеты:
   - Сохраняй максимальный контекст
   - Напиши как можно больше смысловых нод, не пропускай информацию. Ты должен создать максимально полный граф, по максимуму упомянуть все смысловые важные сущности, которые встречаются в тексте. Не теряй информацию.
   - Используй ТОЛЬКО смысловые ноды. Не указывай общеупотребимые ноды (к примеру, сущность камин может быть упомянута, только если это какой-то особенный камин). При этом все именные сущности, все персонажи, значимые места, важные элементы сюжета должны быть упомянуты и описаны.
   - Не выдумывай информацию, которой нет

### Выходной формат:
{{
  "nodes": [
    {{
      "main_name": str,
      "alt_names": list[str],
      "actions": [
        {{
          "source_id": int,
          "chapter_id": int,
          "description": str
        }}
      ],
      "classification": str
    }}
  ],
  "summarization": str
}}
"""


class EntityClassification(str, Enum):
    PERSON = CLASS_CHARACTER
    PLACE = CLASS_PLACE
    ORG = CLASS_ORG
    TERM = CLASS_TERM
    FORCE = CLASS_FORCE


class EvidenceItem(BaseModel):
    source_id: int = Field(..., description="Source chapter id")
    chapter_id: int = Field(..., description="Chapter id")
    description: str = Field(..., min_length=3)

    @validator("description")
    def _validate_description(cls, value: str) -> str:
        val = (value or "").strip()
        if len(val) < 3:
            raise ValueError("description must contain at least 3 chars")
        return val


class ExtractedNode(BaseModel):
    main_name: str = Field(..., min_length=1)
    alt_names: List[str] = Field(default_factory=list)
    actions: List[EvidenceItem] = Field(default_factory=list)
    classification: EntityClassification

    @validator("main_name")
    def _validate_main_name(cls, value: str) -> str:
        val = (value or "").strip()
        if not val:
            raise ValueError("main_name cannot be empty")
        return val

    @validator("alt_names", each_item=True)
    def _validate_alt_names(cls, value: str) -> str:
        return (value or "").strip()


class ExtractedRelation(BaseModel):
    source_node_id: str = Field(..., min_length=1)
    source_node_type: str = Field(..., min_length=1)
    target_node_id: str = Field(..., min_length=1)
    target_node_type: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    descriptions: List[EvidenceItem] = Field(default_factory=list)


class ExtractionPayload(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
    summarization: str = Field(default="")


class ExtractNames(LLMBase):
    """Extractor with pydantic schema."""

    def __init__(self, llm, parser: Optional[PydanticOutputParser] = None):
        if parser is None:
            parser = PydanticOutputParser(pydantic_object=ExtractionPayload)
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            parser=parser,
            parse_json=True,
        )

    @staticmethod
    def _to_source_ids(source_id: Any) -> List[int]:
        if source_id is None:
            return []
        if isinstance(source_id, int):
            return [source_id]
        if isinstance(source_id, (tuple, list)):
            return [int(x) for x in source_id if isinstance(x, int)]
        return []

    @staticmethod
    def _render_chapter_blocks(chapter_blocks: Sequence[Dict[str, Any]]) -> str:
        rendered: List[str] = []
        for block in chapter_blocks:
            sid = block.get("source_id")
            cid = block.get("chapter_id", sid)
            text = (block.get("text") or "").strip()
            if sid is None or not text:
                continue
            rendered.append(
                f"[CHAPTER source_id={sid} chapter_id={cid}]\\n{text}\\n[/CHAPTER]"
            )
        return "\\n\\n".join(rendered)

    def make_user_prompt(
        self,
        text: str,
        source_id: Any = None,
        chapter_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_ids = self._to_source_ids(source_id)
        text_payload = (
            self._render_chapter_blocks(chapter_blocks) if chapter_blocks else text
        )

        if source_ids:
            user_prompt = f"source_ids: {source_ids}\\nТекст: {text_payload}"
        else:
            user_prompt = f"Текст: {text_payload}"
        return {"messages": [("user", user_prompt)]}
