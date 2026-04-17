"""
Промпт и агент для поиска и обработки информации из графа знаний.

Модуль содержит промпт и класс RetrieveAgent для поиска информации
в векторном хранилище и графе знаний, а также формирования ответа.
"""

import logging
import time
from typing import Any, Dict, Optional, Set, Tuple, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.agent.vector_store.retriever import retriever, high_recall_search
from src.agent.vector_store.chapter_retriever import ChapterRetriever
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookGraph, book_graph
from src.utils.graph_search import search_entity_chapter_index, traverse_relations_for_entities
from src.utils.chapter_search import guided_deterministic_search
from src.agent.prompts.output_models import RetrieveOutput
from src.config import settings

logger = logging.getLogger(__name__)

question_answer_prompt = """Ты — эксперт-аналитик, формирующий детальный и подробный ответ на вопрос пользователя по книге. 

# ЗАДАЧА:
Дать ответ ИСКЛЮЧИТЕЛЬНО на основе:
1. Исходный вопрос пользователя: str //Исходный вопрос, который задал пользователь
3. Собранный контекст: List[str] // Уже имеющийся у тебя контекст. Контекст состоит из:
    - Цитаты из книги (небольшие фрагменты из векторного поиска)
    - Суммаризации глав
    - Полные тексты глав (если доступны) - используются для более глубокого понимания контекста
    - Описание действий персонажей/объектов книги, поданные в формате **Действия персонажа** - описание действий персонажа
    - Описание взаимодействия между разными объектами книги, поданные в формате **Взаимодействие между ___ и ___** - описание взаимодействия между персонажами 

# ИНСТРУКЦИИ:

1. Анализ:
   - Сопоставь каждый элемент контекста с запросом
   - Выяви ключевые доказательные факты
   - Опирайся только на ту информацию в контексте, которая имеет отношение к исходному запросу пользователя, не используй нерелевантную информацию

2. Формирование ответа:
   - Структурируй ответ: тезис → доказательства → вывод
   - Используй ТОЧНЫЕ цитаты из контекста
   - Сохрани цепочку логических умозаключений
   - Напиши как можно более подробный ответ, опирающийся на контекст. Не упускай детали, опиши все максимально подробно.

3. Контроль качества:
   - Если информации НЕДОСТАТОЧНО → явно укажи это
   - Запрещены домыслы/интерпретации
   - Ответ должен покрывать ВСЕ аспекты запроса, быть максимально подробным, структурированным и точным. Постарайся ничего не упустить. 

# ПРИМЕРЫ:
Запрос: "Каковы мотивы главного героя в романе X?"
Правильный ответ: {{
  "answer": "Согласно анализу главных диалогов (с. 45-47) и авторским комментариям (с. 112):\n1. Основной мотив - ...\n2. Второстепенный мотив - ...\nВывод: Совокупность факторов указывает на..." 
}}

Недопустимый ответ: "Герой хотел добиться успеха, возможно из-за детских травм" 

# ТРЕБОВАНИЯ К ВЫВОДУ:
- ТОЛЬКО JSON-объект с ключом "answer"
- Ответ на языке оригинала запроса
- Четкая структура с указанием источников
- Минимум 3 доказательных пункта при наличии данных
- Объем: 150-300 слов

###
Ты должен вернуть ответ в следующем формате JSON:

{
  "answer": str // Детальный ответ (минимум 20 символов). Если информации недостаточно, явно укажи это.
}

# ТРЕБОВАНИЯ К ВАЛИДАЦИИ:
- answer: минимум 20 символов
- Ответ должен быть структурированным и содержать конкретную информацию
- Если информации недостаточно, явно укажи это в ответе

{{format_instructions}}

Сформируй ответ:"""


class RetrieveAgent(LLMBase):
    """
    Агент для поиска информации в векторном хранилище и графе знаний.
    
    Выполняет поиск по запросу пользователя, обрабатывает результаты
    из графа знаний (узлы и отношения) и формирует структурированный ответ.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        system_prompt: str,
        retriever: BaseRetriever,
        book_graph: BookGraph,
        parser: Optional[BaseOutputParser] = None,
        load_template: bool = False,
    ):
        """
        Инициализирует RetrieveAgent.
        
        Args:
            llm: Инициализированная языковая модель
            system_prompt: Базовый системный промпт
            retriever: Ретривер для поиска в векторном хранилище
            book_graph: Граф знаний книги
            parser: Парсер для вывода (рекомендуется PydanticOutputParser с RetrieveOutput)
            load_template: Загружать ли промпт из файла
        """
        self.llm = llm
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._prompt_template: Optional[ChatPromptTemplate] = None
        self._parser = parser
        self._book_graph = book_graph
        self._load_template = load_template

    def process_nodes(self, nodes_names: Set[str]) -> str:
        """
        Обрабатывает узлы графа и формирует текстовое описание их действий.
        
        Args:
            nodes_names: Множество имен узлов для обработки
            
        Returns:
            Текстовое описание действий всех узлов
            
        Raises:
            KeyError: Если узел не найден в графе
        """
        all_node_texts = []
        
        for node_name in nodes_names:
            try:
                if node_name not in self._book_graph.nodes.nodes:
                    logger.warning(f"Node '{node_name}' not found in graph, skipping")
                    continue
                    
                node = self._book_graph.nodes.nodes[node_name]
                actions = [action.action for action in node.actions if hasattr(action, 'action')]
                
                node_text = (
                    f"**Действия {node_name}**\n"
                    + "\n".join(actions)
                    + "\n****"
                )
                all_node_texts.append(node_text)
                
            except Exception as e:
                logger.error(f"Error processing node '{node_name}': {e}", exc_info=True)
                continue
        
        return "\n\n".join(all_node_texts)

    def process_rels(self, rel_names: Set[Tuple[str, str]]) -> str:
        """
        Обрабатывает отношения графа и формирует текстовое описание взаимодействий.
        
        Args:
            rel_names: Множество кортежей (имя_узла1, имя_узла2) для обработки
            
        Returns:
            Текстовое описание всех отношений
            
        Raises:
            KeyError: Если отношение не найдено в графе
        """
        all_rels_texts = []
        
        for rel_name in rel_names:
            try:
                if rel_name not in self._book_graph.relationships.relationships:
                    logger.warning(f"Relation '{rel_name}' not found in graph, skipping")
                    continue
                    
                rel = self._book_graph.relationships.relationships[rel_name]
                object_1 = rel.object_1
                object_2 = rel.object_2
                
                descriptions = [
                    desc.description 
                    for desc in rel.description 
                    if hasattr(desc, 'description')
                ]
                desc = "\n".join(descriptions) + "\n****"
                
                rel_text = (
                    f"**Взаимодействие между {object_1} и {object_2}**\n{desc}"
                )
                all_rels_texts.append(rel_text)
                
            except Exception as e:
                logger.error(f"Error processing relation '{rel_name}': {e}", exc_info=True)
                continue
        
        return "\n\n".join(all_rels_texts)

    def process_graph(
        self, 
        rels_result: List[Document], 
        nodes_result: List[Document]
    ) -> Dict[str, str]:
        """
        ???????????? ?????????? ?????? ?? ????? ??????.

        ?????????? ????????? ????? ?? ????? ? ??????????.
        """
        # Primary path: use retrieved graph docs directly.
        # This prevents false "not found in graph, skipping" when vector docs
        # and in-memory graph keys diverge by normalization/version.
        try:
            node_chunks = [
                str(doc.page_content).strip()
                for doc in nodes_result
                if hasattr(doc, "page_content") and str(doc.page_content).strip()
            ]
            rel_chunks = [
                str(doc.page_content).strip()
                for doc in rels_result
                if hasattr(doc, "page_content") and str(doc.page_content).strip()
            ]
            if node_chunks or rel_chunks:
                return {
                    "nodes_texts": "\n***\n".join(node_chunks),
                    "rel_texts": "\n***\n".join(rel_chunks),
                }
        except Exception as e:
            logger.warning("Direct graph-doc formatting failed: %s", e, exc_info=True)

        # Fallback path: legacy lookup by metadata names in in-memory graph.
        try:
            nodes_names: Set[str] = set()
            for doc in nodes_result:
                if hasattr(doc, 'metadata') and "name" in doc.metadata:
                    name = doc.metadata["name"]
                    if isinstance(name, str):
                        nodes_names.add(name)

            rel_names: Set[Tuple[str, str]] = set()
            for doc in rels_result:
                if hasattr(doc, 'metadata') and "name" in doc.metadata:
                    name = doc.metadata["name"]
                    if isinstance(name, tuple) and len(name) == 2:
                        rel_names.add(name)
                    elif isinstance(name, (list, tuple)) and len(name) >= 2:
                        rel_names.add((name[0], name[1]))

            for rel_name in rel_names:
                nodes_names.add(rel_name[0])
                nodes_names.add(rel_name[1])

            node_text = self.process_nodes(nodes_names)
            rels_text = self.process_rels(rel_names)
            return {"nodes_texts": node_text, "rel_texts": rels_text}

        except Exception as e:
            logger.error(f"Error processing graph results: {e}", exc_info=True)
            return {"nodes_texts": "", "rel_texts": ""}

    def _classify_search_docs(self, search_result: List[Document]) -> Dict[str, List[Document]]:
        return {
            "quote_result": [
                doc for doc in search_result
                if hasattr(doc, "metadata") and doc.metadata.get("source") == "book"
            ],
            "nodes_result": [
                doc for doc in search_result
                if hasattr(doc, "metadata") and doc.metadata.get("source") == "nodes"
            ],
            "rels_result": [
                doc for doc in search_result
                if hasattr(doc, "metadata") and doc.metadata.get("source") == "relations"
            ],
            "sums_result": [
                doc for doc in search_result
                if hasattr(doc, "metadata") and doc.metadata.get("source") == "summary"
            ],
        }

    def _format_phase_b_context(self, phase_b: Dict[str, Any]) -> Dict[str, str]:
        entities_text = "\n***\n".join(
            [
                f"{row.get('main_name', '')}: {row.get('summary', '')}"
                for row in (phase_b.get("entities") or [])
            ]
        )
        relations_text = "\n***\n".join(
            [
                (
                    f"{row.get('source_entity', '')} -[{row.get('type', '')}]-> "
                    f"{row.get('target_entity', '')}: {row.get('description', '')}"
                )
                for row in (phase_b.get("relations") or [])
            ]
        )
        chapters_text = "\n***\n".join(
            [
                (
                    f"Глава {row.get('chapter_id', 'N/A')} "
                    f"({row.get('title', '')}): {row.get('snippet', '')}"
                )
                for row in (phase_b.get("chapters") or [])
            ]
        )
        return {
            "entities_text": entities_text,
            "relations_text": relations_text,
            "chapters_text": chapters_text,
        }

    def _collect_source_ids(self, docs: List[Document]) -> Set[int]:
        source_ids: Set[int] = set()
        for doc in docs:
            if hasattr(doc, "metadata"):
                source_id = doc.metadata.get("source_id")
                chapter_id = doc.metadata.get("chapter_id")
                if isinstance(source_id, int):
                    source_ids.add(source_id)
                if isinstance(chapter_id, int):
                    source_ids.add(chapter_id)
        return source_ids

    def tool_semantic_search(
        self,
        query: str,
        hints: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        k_value = int(k or settings.PHASE_A_RECALL_K)
        phase_a = high_recall_search(query=query, k=k_value)
        search_result: List[Document] = phase_a.get("documents") or []
        phase_a_hints: Dict[str, Any] = phase_a.get("hints") or {}
        classified = self._classify_search_docs(search_result)
        quote_result = classified["quote_result"]
        nodes_result = classified["nodes_result"]
        rels_result = classified["rels_result"]
        sums_result = classified["sums_result"]

        graph_ctx = self.process_graph(rels_result, nodes_result)
        source_ids = sorted(self._collect_source_ids(search_result))

        return {
            "tool_name": "semantic_search",
            "input": {"query": query, "hints": hints or {}, "k": k_value},
            "selected_items": phase_a_hints.get("selected_items", []),
            "rejected_items": [],
            "hints": {
                "chapter_ids": phase_a_hints.get("chapter_ids", []),
                "source_ids": phase_a_hints.get("source_ids", []),
                "entity_ids": phase_a_hints.get("entity_ids", []),
                "entity_name_tokens": phase_a_hints.get("entity_name_tokens", []),
            },
            "context": {
                "nodes_texts": graph_ctx.get("nodes_texts", ""),
                "rel_texts": graph_ctx.get("rel_texts", ""),
                "quote_texts": "\n***\n".join(
                    [
                        f"{doc.page_content}. Глава {doc.metadata.get('source_id', doc.metadata.get('chapter_id', 'N/A'))}"
                        for doc in quote_result
                    ]
                ),
                "sums_texts": "\n***\n".join(
                    [
                        f"{doc.page_content}. Глава {doc.metadata.get('source_id', doc.metadata.get('chapter_id', 'N/A'))}"
                        for doc in sums_result
                    ]
                ),
                "source_ids": source_ids,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def tool_deterministic_entity_search(
        self,
        query: str,
        hints: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        hints = hints or {}
        top_k_value = int(top_k or settings.PHASE_B_MAX_ENTITIES)
        index_hints = search_entity_chapter_index(
            query=query,
            hinted_entity_ids=hints.get("entity_ids", []),
            top_k=top_k_value,
        )
        return {
            "tool_name": "deterministic_entity_search",
            "input": {"query": query, "hints": hints, "top_k": top_k_value},
            "selected_items": index_hints.get("selected_items", []),
            "rejected_items": index_hints.get("rejected_items", []),
            "hints": {
                "chapter_ids": index_hints.get("chapter_ids", []),
                "entity_ids": [row.get("entity_id") for row in index_hints.get("entities", []) if row.get("entity_id")],
                "entity_names": [row.get("main_name") for row in index_hints.get("entities", []) if row.get("main_name")],
            },
            "context": {
                "entities_text": "\n***\n".join(
                    [f"{row.get('main_name', '')}: {row.get('summary', '')}" for row in index_hints.get("entities", [])]
                )
            },
            "reason": index_hints.get("reason", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def tool_deterministic_graph_expand(
        self,
        query: str,
        hints: Optional[Dict[str, Any]] = None,
        depth: int = 1,
        relation_priority: Optional[List[str]] = None,
        limits: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        hints = hints or {}
        limits = limits or {}
        max_entities = int(limits.get("max_entities", settings.PHASE_B_MAX_ENTITIES))
        max_relations = int(limits.get("max_relations", settings.PHASE_B_MAX_RELATIONS))
        max_chapters = int(limits.get("max_chapters", settings.PHASE_B_MAX_CHAPTERS))
        phase_b = guided_deterministic_search(
            query=query,
            input_hints={
                "chapter_ids": hints.get("chapter_ids", []),
                "source_ids": hints.get("source_ids", []),
                "entity_ids": hints.get("entity_ids", []),
                "entity_name_tokens": hints.get("entity_name_tokens", []),
            },
            max_entities=max_entities,
            max_relations=max_relations,
            max_chapters=max_chapters,
        )

        seed_entity_names = hints.get("entity_names") or [
            row.get("main_name", "")
            for row in (phase_b.get("entities") or [])
        ]
        traversal = traverse_relations_for_entities(
            entity_names=[name for name in seed_entity_names if isinstance(name, str) and name],
            max_edges=max_relations,
        )
        existing_rel_ids = {
            row.get("relation_id")
            for row in (phase_b.get("relations") or [])
            if row.get("relation_id")
        }
        for rel in traversal.get("selected_items", []):
            rel_id = rel.get("relation_id")
            if rel_id and rel_id in existing_rel_ids:
                continue
            phase_b.setdefault("relations", []).append(rel)

        context = self._format_phase_b_context(phase_b)
        return {
            "tool_name": "deterministic_graph_expand",
            "input": {
                "query": query,
                "hints": hints,
                "depth": depth,
                "relation_priority": relation_priority or [],
                "limits": limits,
            },
            "selected_items": {
                "entities": phase_b.get("entities", []),
                "relations": phase_b.get("relations", []),
                "chapters": phase_b.get("chapters", []),
            },
            "rejected_items": phase_b.get("rejected_items", {}),
            "hints": {
                "chapter_ids": phase_b.get("chapter_ids", []),
                "entity_ids": [row.get("entity_id") for row in (phase_b.get("entities") or []) if row.get("entity_id")],
                "entity_names": [row.get("main_name") for row in (phase_b.get("entities") or []) if row.get("main_name")],
            },
            "context": context,
            "reason": phase_b.get("reason", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def tool_chapter_lookup(
        self,
        chapter_ids: Optional[List[int]] = None,
        max_chapters: Optional[int] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        chapter_ids = [int(v) for v in (chapter_ids or []) if isinstance(v, int)]
        max_chapters_value = int(max_chapters or settings.MAX_CHAPTERS_IN_CONTEXT)
        chapter_retriever = ChapterRetriever()
        chapters_result = chapter_retriever.get_guided_chapters(
            hint_chapter_ids=chapter_ids,
            max_chapters=max_chapters_value,
        )
        chapters = chapters_result.get("chapters", {})
        chapters_text = self._format_chapters_text(
            chapters,
            max_tokens_per_chapter=settings.MAX_CHAPTER_TOKENS,
        )
        selected_items = [
            {
                "chapter_id": int(cid),
                "snippet": text[:320],
            }
            for cid, text in chapters.items()
        ]
        return {
            "tool_name": "chapter_lookup",
            "input": {"chapter_ids": chapter_ids, "max_chapters": max_chapters_value},
            "selected_items": selected_items,
            "rejected_items": chapters_result.get("rejected_items", []),
            "hints": {"chapter_ids": chapters_result.get("selected_items", [])},
            "context": {"chapters_text": chapters_text},
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def run_tool(self, tool_name: str, **tool_input: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "semantic_search":
            return self.tool_semantic_search(
                query=str(tool_input.get("query", "")),
                hints=tool_input.get("hints") or {},
                k=tool_input.get("k"),
            )
        if tool_name == "deterministic_entity_search":
            return self.tool_deterministic_entity_search(
                query=str(tool_input.get("query", "")),
                hints=tool_input.get("hints") or {},
                top_k=tool_input.get("top_k"),
            )
        if tool_name == "deterministic_graph_expand":
            return self.tool_deterministic_graph_expand(
                query=str(tool_input.get("query", "")),
                hints=tool_input.get("hints") or {},
                depth=int(tool_input.get("depth", 1)),
                relation_priority=tool_input.get("relation_priority") or [],
                limits=tool_input.get("limits") or {},
            )
        if tool_name == "chapter_lookup":
            return self.tool_chapter_lookup(
                chapter_ids=tool_input.get("chapter_ids") or [],
                max_chapters=tool_input.get("max_chapters"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def invoke(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет поиск и формирует ответ на запрос пользователя.
        
        Args:
            **kwargs: Аргументы для поиска (должен содержать 'query')
            
        Returns:
            Словарь с ключом "answer" содержащим ответ
            
        Raises:
            ValueError: Если query не передан
            RuntimeError: Если поиск или формирование ответа не удалось
        """
        try:
            query = kwargs.get("query")
            if not query or not isinstance(query, str):
                raise ValueError("query must be a non-empty string")

            trace: List[Dict[str, Any]] = []
            chain_input = self.make_user_prompt_retrieve(query=query)

            # Phase A: wide vector high-recall (hints).
            phase_a_started = time.perf_counter()
            phase_a = high_recall_search(query=chain_input, k=settings.PHASE_A_RECALL_K)
            search_result: List[Document] = phase_a.get("documents") or []
            phase_a_hints: Dict[str, Any] = phase_a.get("hints") or {}
            trace.append(
                {
                    "phase": "phase_a",
                    "query": query,
                    "input_hints": {},
                    "selected_items": phase_a_hints.get("selected_items", []),
                    "rejected_items": [],
                    "reason": "vector high-recall retrieval",
                    "latency_ms": round((time.perf_counter() - phase_a_started) * 1000, 2),
                }
            )

            classified = self._classify_search_docs(search_result)
            quote_result = classified["quote_result"]
            nodes_result = classified["nodes_result"]
            rels_result = classified["rels_result"]
            sums_result = classified["sums_result"]

            # Graph context from vector-retrieved graph docs.
            answer_input = self.process_graph(rels_result, nodes_result)

            quote_texts = "\n***\n".join(
                [
                    f"{doc.page_content}. Глава {doc.metadata.get('source_id', doc.metadata.get('chapter_id', 'N/A'))}"
                    for doc in quote_result
                    if hasattr(doc, "page_content")
                ]
            )
            sums_texts = "\n***\n".join(
                [
                    f"{doc.page_content}. Глава {doc.metadata.get('source_id', doc.metadata.get('chapter_id', 'N/A'))}"
                    for doc in sums_result
                    if hasattr(doc, "page_content")
                ]
            )
            answer_input["quote_texts"] = quote_texts
            answer_input["sums_texts"] = sums_texts

            # Phase B: guided deterministic retrieval over snapshot + graph index.
            phase_b_started = time.perf_counter()
            index_hints = search_entity_chapter_index(
                query=query,
                hinted_entity_ids=phase_a_hints.get("entity_ids", []),
                top_k=settings.PHASE_B_MAX_ENTITIES,
            )
            combined_hints = {
                "chapter_ids": sorted(
                    set(phase_a_hints.get("chapter_ids", []))
                    .union(set(index_hints.get("chapter_ids", [])))
                ),
                "source_ids": list(phase_a_hints.get("source_ids", [])),
                "entity_ids": list(phase_a_hints.get("entity_ids", [])),
                "entity_name_tokens": phase_a_hints.get("entity_name_tokens", []),
            }
            phase_b = guided_deterministic_search(
                query=query,
                input_hints=combined_hints,
                max_entities=settings.PHASE_B_MAX_ENTITIES,
                max_relations=settings.PHASE_B_MAX_RELATIONS,
                max_chapters=settings.PHASE_B_MAX_CHAPTERS,
            )
            traversal = traverse_relations_for_entities(
                entity_names=[row.get("main_name", "") for row in (phase_b.get("entities") or [])],
                max_edges=settings.PHASE_B_MAX_RELATIONS,
            )
            # Добавляем relation traversal как дополнительное подтверждение.
            existing_rel_ids = {
                row.get("relation_id") for row in (phase_b.get("relations") or []) if row.get("relation_id")
            }
            for rel in traversal.get("selected_items", []):
                rel_id = rel.get("relation_id")
                if rel_id and rel_id in existing_rel_ids:
                    continue
                (phase_b.setdefault("relations", [])).append(rel)

            trace.append(
                {
                    "phase": "phase_b",
                    "query": query,
                    "input_hints": combined_hints,
                    "selected_items": {
                        "entities": phase_b.get("entities", []),
                        "relations": phase_b.get("relations", []),
                        "chapters": phase_b.get("chapters", []),
                    },
                    "rejected_items": phase_b.get("rejected_items", {}),
                    "reason": "guided deterministic retrieval using snapshot + entity index",
                    "latency_ms": round((time.perf_counter() - phase_b_started) * 1000, 2),
                }
            )

            phase_b_context = self._format_phase_b_context(phase_b)
            if phase_b_context["entities_text"]:
                answer_input["nodes_texts"] = (
                    (answer_input.get("nodes_texts", "") + "\n\n" + phase_b_context["entities_text"]).strip()
                )
            if phase_b_context["relations_text"]:
                answer_input["rel_texts"] = (
                    (answer_input.get("rel_texts", "") + "\n\n" + phase_b_context["relations_text"]).strip()
                )

            # Phase C: fallback vector pass, if deterministic evidence is weak.
            phase_c_docs: List[Document] = []
            need_phase_c = not phase_b.get("entities") and not phase_b.get("relations")
            phase_c_started = time.perf_counter()
            if need_phase_c:
                phase_c = high_recall_search(query=chain_input, k=settings.PHASE_C_FALLBACK_K)
                phase_c_docs = phase_c.get("documents") or []
                phase_c_hints = phase_c.get("hints") or {}
                phase_c_classified = self._classify_search_docs(phase_c_docs)

                add_quotes = "\n***\n".join(
                    [
                        f"{doc.page_content}. Глава {doc.metadata.get('source_id', doc.metadata.get('chapter_id', 'N/A'))}"
                        for doc in phase_c_classified["quote_result"]
                        if hasattr(doc, "page_content")
                    ]
                )
                if add_quotes:
                    answer_input["quote_texts"] = (answer_input.get("quote_texts", "") + "\n***\n" + add_quotes).strip()

                trace.append(
                    {
                        "phase": "phase_c",
                        "query": query,
                        "input_hints": combined_hints,
                        "selected_items": phase_c_hints.get("selected_items", []),
                        "rejected_items": [],
                        "reason": "fallback vector retrieval because deterministic evidence was weak",
                        "latency_ms": round((time.perf_counter() - phase_c_started) * 1000, 2),
                    }
                )
            else:
                trace.append(
                    {
                        "phase": "phase_c",
                        "query": query,
                        "input_hints": combined_hints,
                        "selected_items": [],
                        "rejected_items": [],
                        "reason": "skipped: deterministic evidence is sufficient",
                        "latency_ms": round((time.perf_counter() - phase_c_started) * 1000, 2),
                    }
                )

            # Chapters for grounding context (guided by A/B hints).
            source_ids = self._collect_source_ids(quote_result + sums_result + phase_c_docs)
            source_ids.update(phase_b.get("chapter_ids") or [])
            chapters_texts = ""
            if settings.INCLUDE_FULL_CHAPTERS and source_ids:
                try:
                    chapter_retriever = ChapterRetriever()
                    chapters_result = chapter_retriever.get_guided_chapters(
                        hint_chapter_ids=sorted(source_ids),
                        max_chapters=settings.MAX_CHAPTERS_IN_CONTEXT,
                    )
                    chapters_texts = self._format_chapters_text(
                        chapters_result.get("chapters", {}),
                        max_tokens_per_chapter=settings.MAX_CHAPTER_TOKENS,
                    )
                except Exception as e:
                    logger.warning(f"Не удалось загрузить главы: {e}", exc_info=True)
                    chapters_texts = ""
            answer_input["chapters_texts"] = chapters_texts
            if phase_b_context["chapters_text"]:
                answer_input["chapters_texts"] = (
                    (answer_input.get("chapters_texts", "") + "\n\n" + phase_b_context["chapters_text"]).strip()
                )

            # Grounding trace before generation.
            evidence_refs = []
            evidence_refs.extend(
                [{"type": "entity", "entity_id": row.get("entity_id")} for row in (phase_b.get("entities") or [])]
            )
            evidence_refs.extend(
                [{"type": "relation", "relation_id": row.get("relation_id")} for row in (phase_b.get("relations") or [])]
            )
            evidence_refs.extend(
                [{"type": "chapter", "chapter_id": row.get("chapter_id")} for row in (phase_b.get("chapters") or [])]
            )
            trace.append(
                {
                    "phase": "grounding",
                    "query": query,
                    "input_hints": {"source_ids": sorted(source_ids)},
                    "selected_items": evidence_refs,
                    "rejected_items": [],
                    "reason": "final evidence set passed to generator",
                    "latency_ms": 0.0,
                    "evidence_refs": evidence_refs,
                }
            )

            try:
                answer_input_query = self.make_user_prompt(query, answer_input)
                result = self.make_llm_chain().invoke(answer_input_query)
                output = self._process_output(result)
                if isinstance(output, dict):
                    output["trace"] = trace
                    return output
                if hasattr(output, "answer"):
                    return {"answer": getattr(output, "answer", ""), "trace": trace}
                if isinstance(output, str):
                    return {"answer": output, "trace": trace}
                logger.warning(f"Unexpected output type: {type(output)}")
                return {"answer": str(output), "trace": trace}
            except Exception as e:
                logger.error(f"Error in LLM chain: {e}", exc_info=True)
                return {"answer": "Ошибка при формировании ответа. Попробуйте переформулировать вопрос.", "trace": trace}

        except ValueError as e:
            logger.error(f"Validation error in invoke: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in invoke: {e}", exc_info=True)
            return {"answer": "Произошла неожиданная ошибка при обработке запроса.", "trace": []}

    def make_user_prompt_retrieve(self, query: str) -> str:
        """
        Форматирует запрос для ретривера.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Отформатированный запрос
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        return query

    def make_user_prompt(self, query: str, answer_input: Dict[str, str]) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для генерации ответа.
        
        Args:
            query: Исходный вопрос пользователя
            answer_input: Словарь с обработанными данными из графа и поиска
            
        Returns:
            Словарь с сообщениями для LLM в формате {"messages": [("user", prompt)]}
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        
        if not isinstance(answer_input, dict):
            answer_input = {}
        
        message = [f"**Исходный вопрос пользователя:** {query}"]
        
        if answer_input.get("nodes_texts"):
            message.extend([
                "==== Описание выбранных персонажей книги: ====",
                answer_input["nodes_texts"],
            ])

        if answer_input.get("rel_texts"):
            message.extend([
                "==== Описание взаимодействия персонажей книги: ====",
                answer_input["rel_texts"],
            ])

        if answer_input.get("sums_texts"):
            message.extend([
                "==== Суммаризация глав книги: ====",
                answer_input["sums_texts"]
            ])

        if answer_input.get("quote_texts"):
            message.extend([
                "==== Цитаты из книги: ====",
                answer_input["quote_texts"]
            ])

        if answer_input.get("chapters_texts"):
            message.extend([
                "==== Полные тексты глав книги: ====",
                answer_input["chapters_texts"]
            ])

        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}
        
        logger.debug(f"Formatted prompt for answer generation (query length: {len(query)})")
        return messages

    def _format_chapters_text(
        self, 
        chapters_dict: Dict[int, str],
        max_tokens_per_chapter: Optional[int] = None
    ) -> str:
        """
        Форматирует тексты глав для добавления в промпт.
        
        Args:
            chapters_dict: Словарь {source_id: chapter_text}
            max_tokens_per_chapter: Максимальное количество токенов на главу
            
        Returns:
            Отформатированная строка с текстами глав
        """
        if not chapters_dict:
            return ""
        
        # Импортируем функцию подсчета токенов
        from tools.preprocess_book.load_to_vectorstore.chunk_splitter import calculate_tokens
        
        formatted_chapters = []
        for source_id, chapter_text in sorted(chapters_dict.items()):
            # Обрезаем главу по токенам если слишком длинная
            text_to_add = chapter_text
            if max_tokens_per_chapter:
                chapter_tokens = calculate_tokens(chapter_text)
                if chapter_tokens > max_tokens_per_chapter:
                    # Обрезаем текст (приблизительно, сохраняя начало)
                    # Используем простую обрезку по символам (4 символа на токен)
                    max_chars = max_tokens_per_chapter * 4
                    text_to_add = chapter_text[:max_chars] + "\n[... текст обрезан ...]"
                    logger.debug(f"Глава {source_id} обрезана: {chapter_tokens} -> {max_tokens_per_chapter} токенов")
            
            formatted_chapters.append(
                f"==== Глава {source_id} ====\n{text_to_add}"
            )
        
        return "\n\n".join(formatted_chapters)


# Создаем парсер с Pydantic моделью для валидации
retrieve_parser = PydanticOutputParser(pydantic_object=RetrieveOutput)

# Обновляем промпт с инструкциями по форматированию
question_answer_prompt_with_format = question_answer_prompt.replace(
    "{{format_instructions}}",
    retrieve_parser.get_format_instructions()
)

retrieve_agent = RetrieveAgent(
    llm,
    question_answer_prompt_with_format,
    retriever=retriever,
    book_graph=book_graph,
    parser=retrieve_parser,
)
