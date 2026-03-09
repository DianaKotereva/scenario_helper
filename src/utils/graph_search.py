import logging
import pickle
import src.config.settings as settings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Action:
    action: str
    source_id: Tuple[int]


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
