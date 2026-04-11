from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.llm_core.llm_prompt_base import LLMBase
from tools.preprocess_book.prompts.extract_names import EvidenceItem, ExtractedRelation

system_prompt = """Ты профессиональный аналитик художественного текста.
Твоя задача: извлечь ТОЛЬКО relations по уже найденным сущностям.

### Входные данные:
- Текст - отрывок главы книги или несколько глав с разметкой [CHAPTER source_id=... chapter_id=...].
- Список нод - список нод, которые уже извлечены из текста. Твоя задача - описания связи между этими нодами.

### Твоя задача:
1. Извлеки отношения как ребра графа:
- source/target_node_id: СТРОГО main_name связанных сущностей
- type: глагол действия в формате "СОВЕРШАЕТ_ДЕЙСТВИЕ" (на русском)
- descriptions: полный контекст взаимодействия в виде СПИСКА объектов:
    {{"source_id": int, "chapter_id": int, "description": str}}
- Учитывай как прямые, так и косвенные связи через события

### Правила обработки:
Важно:
1. Работай строго на русском языке.
2. Не добавляй новые узлы/сущности — используй только список known_nodes из входа.
3. Если связь не подтверждается текстом — не добавляй ее.
4. source_node_id и target_node_id должны быть main_name из known_nodes.
5. Если ты указываешь какого-то персонажа в relations у другого, проверь, что для него есть нода. 
6. Наиболее подробно распиши взаимосвязи между сущностями, напиши максимально подробные отношения между ними - наиболее подробные relations
7. Для конфликтов и трансформаций создавай отдельные связи с разными type
8. descriptions заполняй списком объектов:
   {"source_id": int, "chapter_id": int, "description": str}
9. Разрешены только осмысленные связи между разными сущностями.
10. Не выдумывай информацию, которой нет

Формат ответа:
{
  "relations": [
    {
      "source_node_id": str,
      "source_node_type": str,
      "target_node_id": str,
      "target_node_type": str,
      "type": str,
      "descriptions": [
        {
          "source_id": int,
          "chapter_id": int,
          "description": str
        }
      ]
    }
  ]
}
"""


class RelationsPayload(BaseModel):
    relations: List[ExtractedRelation] = Field(default_factory=list)


class ExtractRelations(LLMBase):
    """Second-stage extractor: text + known nodes -> relations only."""

    def __init__(self, llm, parser: Optional[PydanticOutputParser] = None):
        if parser is None:
            parser = PydanticOutputParser(pydantic_object=RelationsPayload)
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
                f"[CHAPTER source_id={sid} chapter_id={cid}]\n{text}\n[/CHAPTER]"
            )
        return "\n\n".join(rendered)

    @staticmethod
    def _prepare_known_nodes(
        known_nodes: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        prepared: List[Dict[str, str]] = []
        for item in known_nodes:
            main_name = str(item.get("main_name", "")).strip()
            classification = str(item.get("classification", "")).strip()
            if not main_name:
                continue
            prepared.append(
                {
                    "main_name": main_name,
                    "classification": classification or "неизвестно",
                }
            )
        return prepared

    def make_user_prompt(
        self,
        text: str,
        known_nodes: Sequence[Dict[str, Any]],
        source_id: Any = None,
        chapter_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_ids = self._to_source_ids(source_id)
        text_payload = (
            self._render_chapter_blocks(chapter_blocks) if chapter_blocks else text
        )
        nodes_payload = self._prepare_known_nodes(known_nodes)

        data = {
            "source_ids": source_ids,
            "known_nodes": nodes_payload,
            "text": text_payload,
        }
        user_prompt = json.dumps(data, ensure_ascii=False)
        return {"messages": [("user", user_prompt)]}
