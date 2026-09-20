from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from langchain_core.output_parsers import PydanticOutputParser

from tools.preprocess_book.prompts.extract_names import ExtractNames, ExtractionPayload


class ExtractMissingNodes(ExtractNames):
    """Specialized extractor for retrying missing entity mentions."""

    def __init__(self, llm, parser: Optional[PydanticOutputParser] = None):
        if parser is None:
            parser = PydanticOutputParser(pydantic_object=ExtractionPayload)
        super().__init__(llm=llm, parser=parser)

    def make_user_prompt(
        self,
        text: str,
        source_id: Any = None,
        chapter_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        missing_mentions: Optional[Sequence[str]] = None,
        existing_nodes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return super().make_user_prompt(
            text=text,
            source_id=source_id,
            chapter_blocks=chapter_blocks,
            required_mentions=missing_mentions,
            existing_nodes=existing_nodes,
            retry_mode="add_missing_only",
        )

