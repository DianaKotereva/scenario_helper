"""
Ретривер для поиска в векторном хранилище.

Модуль содержит класс OpenSearchStore для настройки и инициализации
ретривера с поддержкой BM25 и векторного поиска.
"""

import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Set

try:
    from langchain.retrievers import EnsembleRetriever
except Exception:  # pragma: no cover - compatibility for langchain>=1.x
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.agent.vector_store.vector_store import VectorStore
from src.config import settings

logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_entity_name_tokens(doc: Document) -> Set[str]:
    tokens: Set[str] = set()
    if not hasattr(doc, "metadata"):
        return tokens
    name_value = doc.metadata.get("name")
    if isinstance(name_value, str):
        for token in re.split(r"[\s,;:()]+", name_value.lower()):
            t = token.strip()
            if len(t) >= 3:
                tokens.add(t)
    return tokens


def extract_retrieval_hints(documents: List[Document]) -> Dict[str, Any]:
    """Извлекает chapter/entity hints из выдачи ретривера."""
    chapter_ids: Set[int] = set()
    source_ids: Set[int] = set()
    entity_ids: Set[str] = set()
    entity_name_tokens: Set[str] = set()
    by_source: Dict[str, int] = {}
    selected_items: List[Dict[str, Any]] = []

    for idx, doc in enumerate(documents):
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1

        source_id = _safe_int(meta.get("source_id"))
        if source_id is not None:
            source_ids.add(source_id)
            chapter_ids.add(source_id)

        chapter_id = _safe_int(meta.get("chapter_id"))
        if chapter_id is not None:
            chapter_ids.add(chapter_id)
        elif source_id is not None:
            # Для граф-доков chapter_id может отсутствовать, но source_id == chapter_id.
            chapter_id = source_id

        entity_id = meta.get("entity_id")
        if isinstance(entity_id, str) and entity_id.strip():
            entity_ids.add(entity_id.strip())

        entity_name_tokens.update(_extract_entity_name_tokens(doc))

        selected_items.append(
            {
                "rank": idx + 1,
                "source": source,
                "source_id": source_id,
                "chapter_id": chapter_id,
                "entity_id": entity_id if isinstance(entity_id, str) else None,
                "snippet": str(getattr(doc, "page_content", "") or "")[:220],
            }
        )

    return {
        "chapter_ids": sorted(chapter_ids),
        "source_ids": sorted(source_ids),
        "entity_ids": sorted(entity_ids),
        "entity_name_tokens": sorted(entity_name_tokens),
        "by_source": by_source,
        "selected_items": selected_items,
    }


class OpenSearchStore:
    """
    Класс для настройки и управления ретривером OpenSearch.
    
    Объединяет BM25 и векторный поиск в единый ансамблевый ретривер.
    """
    
    def __init__(
        self, 
        documents: Optional[List[Document]] = None, 
        force_reload: bool = False
    ):
        """
        Инициализирует OpenSearchStore.
        
        Args:
            documents: Список документов для индексации (опционально)
            force_reload: Принудительная перезагрузка индекса
            
        Raises:
            FileNotFoundError: Если файл с документами не найден
            ValueError: Если документы не предоставлены и файл не найден
        """
        self.force_reload = force_reload
        self._embeddings = self._setup_embeddings()
        self._vector_store = self._setup_vector_store()
        self._retriever = self._setup_retriever(documents)

    def _setup_embeddings(self) -> OpenAIEmbeddings:
        """
        Настраивает функцию эмбеддингов.
        
        Returns:
            Экземпляр OpenAIEmbeddings
            
        Raises:
            ValueError: Если API ключ не установлен
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in settings")
        
        embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMB_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        logger.debug("Embeddings initialized successfully")
        return embeddings

    def _setup_retriever(self, documents: Optional[List[Document]] = None) -> EnsembleRetriever:
        """
        Настраивает ансамблевый ретривер (BM25 + векторный поиск).
        
        Args:
            documents: Список документов для индексации
            
        Returns:
            Экземпляр EnsembleRetriever
            
        Raises:
            FileNotFoundError: Если файл с документами не найден
            ValueError: Если документы не предоставлены
        """
        pickle_documents_path = settings.ES_PICKLE_DOCUMENTS_PATH
        
        # Загружаем документы если не предоставлены
        if not documents and pickle_documents_path and os.path.exists(pickle_documents_path):
            try:
                with open(pickle_documents_path, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Loaded {len(documents)} documents from {pickle_documents_path}")
            except Exception as e:
                logger.warning(
                    "Could not load BM25 documents from %s, continue with vector-only retrieval. Error: %s",
                    pickle_documents_path,
                    e,
                )
                documents = []
        
        # Настраиваем BM25 ретривер

        bm25_retriever = None
        if documents:
            try:
                bm25_retriever = BM25Retriever.from_documents(
                    documents=documents,
                    k=settings.BM25_K,
                )
                logger.debug(f"BM25 retriever initialized with k={settings.BM25_K}")
            except Exception as e:
                logger.warning(
                    "Error initializing BM25 retriever, continue with vector-only retrieval: %s",
                    e,
                    exc_info=True,
                )
        
        # Настраиваем векторный ретривер
        try:
            es_retriever = self._vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": settings.ES_RETRIEVER_K,
                    "score_threshold": settings.ES_SIMILARITY_THRESHOLD,
                },
            )
            logger.debug(
                f"Vector retriever initialized with k={settings.ES_RETRIEVER_K}, "
                f"threshold={settings.ES_SIMILARITY_THRESHOLD}"
            )
        except Exception as e:
            logger.error(f"Error initializing vector retriever: {e}", exc_info=True)
            raise
        
        # Создаем ансамблевый ретривер

        if bm25_retriever is None:
            logger.info("Vector-only retriever initialized (BM25 disabled)")
            return es_retriever

        try:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[es_retriever, bm25_retriever],
                weights=[0.8, 0.2],  # 80% vector, 20% BM25
            )
            logger.info("Ensemble retriever initialized successfully")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error initializing ensemble retriever: {e}", exc_info=True)
            raise

    def _setup_vector_store(self) -> VectorStore:
        """
        Настраивает векторное хранилище.
        
        Returns:
            Экземпляр VectorStore
        """
        try:
            vector_store = VectorStore(
                embedding_function=self._embeddings,
                index_name=settings.ES_INDEX_NAME,
                batch_size=settings.ES_BATCH_SIZE,
                pickle_documents_path=settings.ES_PICKLE_DOCUMENTS_PATH,
                full_reload=self.force_reload,
                faiss_path=settings.FAISS_PATH,
                setup_index=self.force_reload,
            )
            logger.debug("Vector store initialized successfully")
            return vector_store.store
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}", exc_info=True)
            raise

    def high_recall_search(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Фаза A: широкий high-recall retrieval с возвратом hints.

        Returns:
            dict: {"documents": List[Document], "hints": {...}}
        """
        if not query or not isinstance(query, str):
            return {"documents": [], "hints": extract_retrieval_hints([])}

        target_k = k or settings.PHASE_A_RECALL_K
        docs: List[Document] = []

        # 1) Пытаемся использовать ensemble retriever.
        try:
            docs = self._retriever.invoke(query) or []
            if target_k > 0:
                docs = docs[:target_k]
        except Exception as e:
            logger.warning("High-recall via ensemble retriever failed: %s", e, exc_info=True)

        # 2) Fallback на прямой vector similarity.
        if not docs:
            try:
                docs = self._vector_store.similarity_search(query=query, k=target_k) or []
            except Exception as e:
                logger.warning("High-recall vector fallback failed: %s", e, exc_info=True)
                docs = []

        hints = extract_retrieval_hints(docs)
        return {"documents": docs, "hints": hints}


def high_recall_search(query: str, k: Optional[int] = None) -> Dict[str, Any]:
    """Глобальный helper для Phase A."""
    if elastic_store is None:
        docs = []
        try:
            docs = retriever.invoke(query) if retriever else []
        except Exception as e:
            logger.warning("Fallback high_recall_search failed: %s", e, exc_info=True)
        return {"documents": docs, "hints": extract_retrieval_hints(docs)}
    return elastic_store.high_recall_search(query=query, k=k)


# Создаем глобальный экземпляр ретривера
try:
    elastic_store = OpenSearchStore(force_reload=settings.FORCE_RELOAD)
    retriever = elastic_store._retriever
    logger.info("Global retriever initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize global retriever: {e}", exc_info=True)
    retriever = None
