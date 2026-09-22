from __future__ import annotations

import json
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


system_prompt = f"""<MAIN_TASK_GUIDELINES>
You are a professional text analyst who transforms literary works into detailed semantic graphs.  
Your task: extract named entities from the provided text and output them as a structured JSON graph.

You will receive a text with markup: `[CHAPTER source_id=... chapter_id=...]`  
You must extract all meaningful entities, describe their actions per chapter, and classify each node strictly into one of five categories:  
`"{CLASS_CHARACTER}"`, `"{CLASS_PLACE}"`, `"{CLASS_ORG}"`, `"{CLASS_TERM}"`, `"{CLASS_FORCE}"`.
These classification values are fixed schema labels; do not translate them when the input language changes.

Output format (JSON):
{{
  "nodes": [
    {{
      "main_name": str,
      "alt_names": list[str],
      "actions": [
        {{"source_id": int, "chapter_id": int, "description": str}}
      ],
      "classification": str
    }}
  ]
}}
</MAIN_TASK_GUIDELINES>

<DATE_GUIDELINES>
PAY ATTENTION TO THE DATE INSIDE THE USER REQUEST  
DATE FORMAT: YYYY-MM-DD HH:MM:SS (ISO 8601)  
IMPORTANT: The date above is in YYYY-MM-DD format (Year-Month-Day). For example, 2025-10-03 means October 3rd, 2025, NOT March 10th.  
Note: This task does not primarily depend on dates, but if the text mentions temporal markers, treat them as part of entity descriptions.
</DATE_GUIDELINES>

<IMPORTANT_LANGUAGE_GUIDELINES>
Detect the language of the **input text** (the literary excerpt) and use this language for all extracted string fields:  
`main_name`, every string inside `alt_names`, and every `description` inside `actions`.  
LANGUAGE ADAPTATION: Always respond in the SAME LANGUAGE as the input text.  
If the input is in Russian – extract in Russian; if in English – in English; if in French – in French, etc.  
Do not translate names or descriptions into another language.
</IMPORTANT_LANGUAGE_GUIDELINES>

<CORE_PRINCIPLES>
1. Extract as much information as possible from the text without sacrificing accuracy.  
2. Do not add data that is not explicitly mentioned in the text.  
3. Do not omit any meaningful named entity (characters, places, organizations, unique terms, forces of nature).  
4. Prefer to extract more rather than less, but only from what is actually stated.  
5. Ensure every node has a valid `classification` (one of the five values) – `null` is forbidden.  
6. If an entity fits multiple classes, choose the most relevant one based on context.  
7. For common objects (e.g., a regular fireplace) – include only if they are special or plot‑significant.
</CORE_PRINCIPLES>

<GROUNDING_RULES>
STRICT TEXT GROUNDING (MANDATORY):  
1. Answer ONLY from data explicitly present in the provided input text.  
2. DO NOT use prior/world knowledge "from memory" if it is not confirmed by the input.  
3. DO NOT invent facts, names, relations, or actions.  
4. If the input lacks information about an entity's action or alternative name, leave the corresponding field empty (`[]` for lists).  
5. For `main_name`, choose the most stable canonical name across the whole input (usually the most frequent or first explicit name).  
6. For `alt_names`, include only meaningful alternative variants that actually appear in the text (nicknames, diminutives, descriptive epithets used as names).  
7. For `actions`, create one object per distinct action or role per `(source_id, chapter_id)` block. If the entity is merely mentioned without action, you may describe as "mentioned" or "present".
</GROUNDING_RULES>

<REASONING_GUIDELINES>
ADAPTIVITY: Process the text sequentially; if a later chapter introduces a new name for an already seen entity, add that name to `alt_names` of the same node (same `main_name`).  
ANALYSIS EXTRACT DATA: Always scan for all named entities, including episodic characters and unique terms. Do not skip any meaningful proper noun or well‑defined concept.  
If a pronoun (he/she/it) is used without clear antecedent, try to resolve from recent context; if impossible, do not create a node.
Resolving a pronoun may help assign an action to an entity, but the pronoun itself must never be added to `alt_names`.
</REASONING_GUIDELINES>

<PRECISION_GUIDELINES>
CRITICAL FOR FACTUAL ACCURACY:  
1. EXACT VALUES: For `description`, use exact phrasing from the text where possible, keeping the original language.  
2. SOURCE AND CHAPTER IDs: Copy them exactly as provided in the `[CHAPTER ...]` markers.  
3. NO SPECULATION: If a character is not named, do not invent a name. If a place is not named, do not extract it as a node (unless it is described as unique and significant without a proper name? – better skip).  
4. CROSS‑VERIFICATION: Within the same input, if the same entity's name appears with different spellings, include these observed spelling variants in `alt_names` and keep one `main_name`. An explicit identity statement is not required for spelling variants of the same name. Pronouns are not spelling variants of names and must never be included in `alt_names`, even when their referent is clear.  
5. TEMPORAL VALIDATION: If the text mentions time (e.g., "three years later"), you may include that information in the `description` as stated, but do not alter the meaning.
</PRECISION_GUIDELINES>

<AGENT_TOOL_USAGE_GUIDELINES>
This task does not require external tools. The input text is provided directly in the user request.  
Simply analyze the text as given. No searching, browsing, or API calls are needed.  
If, hypothetically, tools were available, you would ignore them for this specific task.
</AGENT_TOOL_USAGE_GUIDELINES>
<ALIAS_STRICT_RULES>
Spelling variants of the same entity's name follow CROSS-VERIFICATION above;
they do not require an explicit identity statement.
For other names or nicknames, add them to alt_names ONLY if the text explicitly
states identity equivalence (e.g. "X is Y", "also known as", "called/named", direct renaming).
Never add pronouns to alt_names, regardless of whether their referent is known.

For names that are not spelling variants of the same entity's name, do NOT add to alt_names merely because they are:
- in the same sentence/phrase without explicit equivalence,
- linked by relation, title, possession, command, kinship, mount/rider, service,
- co-mentioned in enumeration or apposition,
- used as different agents/patients of actions.

If unsure, keep separate nodes.
</ALIAS_STRICT_RULES>

<CROSS_NODE_CONSTRAINT>
Within one extraction output, a string that appears as main_name of any node
must not be added to alt_names of another node, unless explicit identity
equivalence is stated in text or CROSS-VERIFICATION identifies them as spelling
variants of the same entity's name. In either case, keep one node for that entity.
</CROSS_NODE_CONSTRAINT>

<SELF_CHECK>
Before finalizing JSON:
- verify every alt_name is an observed spelling variant of the same entity's name or has explicit textual identity evidence;
- remove pronouns from alt_names, even when their referent is clear;
- remove any alt_name that can denote a distinct entity in the same input;
- prefer split over merge when identity is ambiguous.
</SELF_CHECK>

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
        required_mentions: Optional[Sequence[str]] = None,
        existing_nodes: Optional[Sequence[Dict[str, Any]]] = None,
        retry_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_ids = self._to_source_ids(source_id)
        text_payload = (
            self._render_chapter_blocks(chapter_blocks) if chapter_blocks else text
        )

        mentions_payload: List[str] = []
        if required_mentions:
            seen: set[str] = set()
            for mention in required_mentions:
                val = str(mention or "").strip()
                if not val:
                    continue
                key = val.lower()
                if key in seen:
                    continue
                seen.add(key)
                mentions_payload.append(val)

        if source_ids:
            user_prompt = f"source_ids: {source_ids}\nText: {text_payload}"
        else:
            user_prompt = f"Text: {text_payload}"

        if retry_mode == "add_missing_only" and mentions_payload:
            user_prompt += (
                "\n\nRetry task: add only missing entities to nodes. "
                "Do not delete or rewrite already extracted entities."
            )

        if existing_nodes:
            compact_nodes = []
            for node in existing_nodes:
                if not isinstance(node, dict):
                    continue
                compact_nodes.append(
                    {
                        "main_name": node.get("main_name", ""),
                        "alt_names": node.get("alt_names", []),
                        "classification": node.get("classification", ""),
                    }
                )
            if compact_nodes:
                user_prompt += (
                    "\n\nAlready extracted nodes (context, do not remove): "
                    f"{json.dumps(compact_nodes, ensure_ascii=False)}"
                )

        if mentions_payload:
            user_prompt += (
                "\n\nRequired mentions for coverage check: "
                f"{mentions_payload}\n"
                "If these entities are truly present in the text, include each of them "
                "at least once in main_name or alt_names. Do not invent entities."
            )

        return {"messages": [("user", user_prompt)]}
