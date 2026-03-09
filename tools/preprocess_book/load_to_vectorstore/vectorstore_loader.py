"""
Модуль для загрузки документов в векторное хранилище.

Содержит класс VectorStoreLoader для сохранения документов
и загрузки их в векторное хранилище (OpenSearch/FAISS).
"""

import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.agent.vector_store.vector_store import VectorStore
from src.config import settings
from tools.preprocess_book.storage.storage import FileManager

logger = logging.getLogger(__name__)


def _stable_doc_id(doc: Document) -> int:
    meta = doc.metadata or {}
    source = str(meta.get("source", "unknown"))
    source_id = str(meta.get("source_id", "unknown"))
    uid = str(meta.get("doc_uid", ""))
    payload = f"{source}|{source_id}|{uid}|{doc.page_content}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


class VectorStoreLoader:
    """
    Класс для загрузки документов в векторное хранилище.
    
    Сохраняет документы в pickle файл и загружает их
    в векторное хранилище через VectorStore.
    """
    
    def __init__(self):
        """Инициализирует VectorStoreLoader."""
        self._embeddings = self._setup_embeddings()
    
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
    
    def save_documents_to_pickle(
        self, 
        documents: List[Document], 
        output_path: Path
    ) -> None:
        """
        Сохраняет документы в pickle файл.
        
        Args:
            documents: Список Document объектов для сохранения
            output_path: Путь к файлу для сохранения
            
        Raises:
            IOError: Если не удается сохранить файл
        """
        try:
            # Создаем директорию если она не существует
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Stable id for deterministic incremental upsert.
            for doc in documents:
                if "source_id" not in doc.metadata:
                    logger.warning(f"Document without source_id: {doc.metadata}")
                doc.metadata["id"] = _stable_doc_id(doc)
            
            # Сохраняем документы
            with open(output_path, "wb") as file:
                pickle.dump(documents, file)
            
            logger.info(f"Сохранено {len(documents)} документов в {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении документов: {e}", exc_info=True)
            raise
    
    def load_documents(
        self,
        documents: List[Document],
        output_pickle_path: Path,
        force_reload: bool = False,
        index_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        changed_source_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Сохраняет документы и загружает их в векторное хранилище.
        
        Args:
            documents: Список Document объектов для загрузки
            output_pickle_path: Путь к pickle файлу для сохранения документов
            force_reload: Принудительная перезагрузка индекса
            index_name: Имя индекса (по умолчанию из settings.ES_INDEX_NAME)
            batch_size: Размер батча (по умолчанию из settings.ES_BATCH_SIZE)
            
        Raises:
            ValueError: Если документы пусты
            RuntimeError: Если не удается загрузить в хранилище
        """
        if not documents:
            raise ValueError("Список документов пуст")
        
        try:
            # Сохраняем документы в pickle
            logger.info(f"Сохранение {len(documents)} документов в pickle...")
            self.save_documents_to_pickle(documents, output_pickle_path)
            
            # Настраиваем параметры VectorStore
            index_name = index_name or settings.ES_INDEX_NAME
            batch_size = batch_size or settings.ES_BATCH_SIZE
            
            # Определяем путь к FAISS (если используется)
            faiss_path = None
            if getattr(settings, "USE_FAISS", False):
                faiss_path = settings.FAISS_PATH
            
            # Создаем VectorStore
            logger.info(f"Инициализация VectorStore с индексом '{index_name}'...")
            vector_store = VectorStore(
                embedding_function=self._embeddings,
                index_name=index_name,
                batch_size=batch_size,
                pickle_documents_path=str(output_pickle_path),
                full_reload=force_reload,
                faiss_path=faiss_path,
                changed_source_ids=changed_source_ids,
            )
            
            logger.info("Документы успешно загружены в векторное хранилище")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке документов в векторное хранилище: {e}", exc_info=True)
            raise
