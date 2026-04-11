from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.llm_core.llm_prompt_base import LLMBase
from tools.preprocess_book.prompts.extract_names import ExtractedRelation

system_prompt = """Ты профессиональный аналитик художественного текста.
Твоя задача: извлечь только relations между уже найденными nodes.

Важно:
- Работай строго на русском языке.
- Не добавляй новые сущности, используй только known_nodes из входа.
- Если связь не подтверждается текстом, не добавляй её.

Для каждой связи верни:
- source_node_id: main_name из known_nodes
- source_node_type: тип source-сущности
- target_node_id: main_name из known_nodes
- target_node_type: тип target-сущности
- type: название отношения
- descriptions: список объектов вида
  {"source_id": int, "chapter_id": int, "description": str}

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
