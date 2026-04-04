"""Data contracts for snapshot export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SNAPSHOT_FILENAMES: Dict[str, str] = {
    "chunks": "chunks.jsonl",
    "chapters": "chapters.jsonl",
    "entities": "entities.jsonl",
    "relations": "relations.jsonl",
    "entity_chapter_index": "entity_chapter_index.jsonl",
}


REQUIRED_FIELDS: Dict[str, List[str]] = {
    "chunks": ["chunk_id", "text", "source_id", "chapter_id", "start", "end", "metadata"],
    "chapters": ["chapter_id", "title", "text", "source_id"],
    "entities": ["entity_id", "main_name", "alt_names", "classification", "record_type"],
    "relations": [
        "relation_id",
        "source_entity",
        "target_entity",
        "type",
        "description",
        "record_type",
    ],
    "entity_chapter_index": ["entity_id", "main_name", "chapters", "aliases", "mentions_count"],
}


@dataclass(slots=True)
class ActionByChapter:
    chapter_id: int
    summary: str
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ChapterRecord:
    chapter_id: int
    title: str
    text: str
    source_id: int


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_id: int
    chapter_id: int
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    main_name: str
    alt_names: List[str]
    classification: str
    record_type: str  # raw | merged
    actions: Optional[List[str]] = None
    chapter_id: Optional[int] = None
    source_id: Optional[int] = None
    chapter_ids: Optional[List[int]] = None
    source_ids: Optional[List[int]] = None
    actions_by_chapter: Optional[List[ActionByChapter]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RelationRecord:
    relation_id: str
    source_entity: str
    target_entity: str
    type: str
    description: str
    record_type: str  # raw | merged
    chapter_id: Optional[int] = None
    source_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityChapterIndexRecord:
    entity_id: str
    main_name: str
    chapters: List[int]
    aliases: List[str]
    mentions_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
