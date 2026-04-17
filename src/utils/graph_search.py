import logging
import pickle
import json
import re
from functools import lru_cache
from pathlib import Path
import src.config.settings as settings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class Action:
    action: str
    source_id: Tuple[int]
    chapter_id: Optional[int] = None
    quotes: Optional[List[str]] = field(default_factory=list)


@dataclass
class BookNode:
    main_name: str  # Имя персонажа или объекта
    classification: str  # Классификация NER. Возможные классы: ['персонаж', 'место', 'организация', 'термин', 'сила природы']
    alt_names: Optional[List[str]] = field(
        default_factory=list
    )  # Альтернативные имена персонажей или мест. К примеру, один и тот же человек может быть по разному назван разными персонажами. Главным именем считается его основное имя, с которым он большую частью сюжета фигурирует. Все остальные имена и прозвища - альтернативные имена.
    actions: Optional[List[Action]] = field(
        default_factory=list
    )  # Краткий пересказ всех действий, которые делал персонаж в книге.


@dataclass
class AllBookNodes:
    nodes: Dict[str, BookNode] = field(default_factory=dict)
    names_list: Dict[str, str] = field(default_factory=dict)


@dataclass
class Description:
    description: str
    type: str
    source_id: Tuple[int]
    chapter_id: Optional[int] = None
    quotes: Optional[List[str]] = field(default_factory=list)


@dataclass
class BookEdges:
    object_1: str  # Объект 1 (имя)
    object_2: str  # Объект 2 (имя)
    description: List[Description]  # Характер связи - незафиксированный класс, абзац


@dataclass
class AllBooksEdges:
    relationships: Dict[Tuple[str], BookEdges]


@dataclass
class BookGraph:
    nodes: AllBookNodes
    relationships: AllBooksEdges


# Кастомный Unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Игнорируем модуль и ищем класс по имени в глобальных переменных
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Class '{name}' not found in globals")


# Загрузка графа с использованием CustomUnpickler
try:
    with open(settings.GRAPH_PICKLE_PATH, "rb") as file:
        unpickler = CustomUnpickler(file)
        book_graph = unpickler.load()
except Exception as e:
    logger.warning(
        "Failed to load graph pickle from %s, using empty graph. Error: %s",
        settings.GRAPH_PICKLE_PATH,
        e,
    )
    book_graph = BookGraph(
        nodes=AllBookNodes(nodes={}, names_list={}),
        relationships=AllBooksEdges(relationships={}),
    )


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {t for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower()) if len(t) >= 3}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _read_first_existing_jsonl(paths: List[Path]) -> List[Dict[str, Any]]:
    for path in paths:
        rows = _read_jsonl(path)
        if rows:
            return rows
    return []


def _resolve_snapshot_dir(snapshot_dir: Optional[str]) -> Path:
    if snapshot_dir:
        return Path(snapshot_dir)

    configured = Path(settings.SNAPSHOT_DIR)
    if configured.exists():
        has_index = bool(_read_jsonl(configured / "entity_chapter_index.jsonl"))
        has_relations = bool(
            _read_first_existing_jsonl(
                [configured / "relations_merged.jsonl", configured / "relations.jsonl"]
            )
        )
        if has_index and has_relations:
            return configured
        logger.warning(
            "Configured SNAPSHOT_DIR=%s is present but empty/incomplete, fallback to latest processed_data snapshot",
            configured,
        )

    processed_data = Path("processed_data")
    if not processed_data.exists():
        return configured

    candidates = sorted(
        [
            d / "snapshot"
            for d in processed_data.iterdir()
            if d.is_dir() and (d / "snapshot").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else configured


@lru_cache(maxsize=2)
def _load_snapshot_indexes(snapshot_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    base = _resolve_snapshot_dir(snapshot_dir)
    return {
        "entity_chapter_index": _read_jsonl(base / "entity_chapter_index.jsonl"),
        "relations_merged": _read_first_existing_jsonl(
            [base / "relations_merged.jsonl", base / "relations.jsonl"]
        ),
    }


def search_entity_chapter_index(
    query: str,
    hinted_entity_ids: Optional[List[str]] = None,
    snapshot_dir: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Поиск кандидатов по entity_chapter_index для guided deterministic retrieval.
    """
    snapshot = _load_snapshot_indexes(snapshot_dir if snapshot_dir else "")
    rows = snapshot["entity_chapter_index"]
    hinted = {x for x in (hinted_entity_ids or []) if isinstance(x, str)}
    q_tokens = _tokenize(query)

    scored: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in rows:
        text = " ".join(
            [str(row.get("main_name", ""))]
            + [str(x) for x in (row.get("aliases") or [])]
        )
        overlap = q_tokens.intersection(_tokenize(text))
        score = float(len(overlap))
        if row.get("entity_id") in hinted:
            score += 2.0
        if score <= 0:
            rejected.append({"entity_id": row.get("entity_id"), "reason": "no_overlap"})
            continue
        scored.append(
            {
                "entity_id": row.get("entity_id"),
                "main_name": row.get("main_name"),
                "chapters": row.get("chapters") or [],
                "aliases": row.get("aliases") or [],
                "mentions_count": row.get("mentions_count", 0),
                "score": score,
            }
        )
    scored.sort(key=lambda x: (x["score"], x.get("mentions_count", 0)), reverse=True)
    selected = scored[:top_k]

    chapter_ids: Set[int] = set()
    for item in selected:
        for cid in item.get("chapters", []):
            if isinstance(cid, int):
                chapter_ids.add(cid)

    return {
        "entities": selected,
        "chapter_ids": sorted(chapter_ids),
        "selected_items": selected,
        "rejected_items": rejected[:30],
        "reason": "entity_chapter_index lookup",
    }


def traverse_relations_for_entities(
    entity_names: List[str],
    snapshot_dir: Optional[str] = None,
    max_edges: int = 25,
) -> Dict[str, Any]:
    """
    Ограниченный traversal по relations_merged:
    оставляем ребра, где хотя бы одна сторона в релевантных сущностях.
    """
    snapshot = _load_snapshot_indexes(snapshot_dir if snapshot_dir else "")
    relations = snapshot["relations_merged"]
    names = {n.lower() for n in entity_names if isinstance(n, str) and n.strip()}

    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for rel in relations:
        src = str(rel.get("source_entity", "")).lower()
        dst = str(rel.get("target_entity", "")).lower()
        if src in names or dst in names:
            selected.append(
                {
                    "relation_id": rel.get("relation_id"),
                    "source_entity": rel.get("source_entity"),
                    "target_entity": rel.get("target_entity"),
                    "type": rel.get("type"),
                    "description": rel.get("description"),
                    "chapter_id": rel.get("chapter_id"),
                }
            )
            if len(selected) >= max_edges:
                break
        else:
            rejected.append({"relation_id": rel.get("relation_id"), "reason": "outside_entity_scope"})

    return {
        "selected_items": selected,
        "rejected_items": rejected[:30],
        "reason": "limited relation traversal",
    }
