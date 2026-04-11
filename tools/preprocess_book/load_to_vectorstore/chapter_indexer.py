"""
Модуль для сохранения глав книги в обычный ElasticSearch индекс.

Содержит класс ChapterIndexer для сохранения полных текстов глав
в отдельный индекс (не векторный) для последующего поиска по source_id.
"""

import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from langchain_core.documents import Document

try:
    from opensearchpy import OpenSearch, helpers
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("OpenSearch not available. Install with: pip install opensearch-py")

from tools.preprocess_book.config.preprocess_settings import (
    CHAPTERS_INDEX_NAME,
    CHAPTERS_PICKLE_PATH,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


class ChapterIndexer:
    """
    Класс для сохранения глав книги в обычный ElasticSearch индекс.
    
    Главы сохраняются в отдельный индекс без векторизации для последующего
    поиска по source_id.
    """
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Инициализирует ChapterIndexer.
        
        Args:
            index_name: Имя индекса (по умолчанию из настроек)
        """
        self.index_name = index_name or CHAPTERS_INDEX_NAME
        self._client = None
        self._load_opensearch_settings()
    
    def _load_opensearch_settings(self) -> Dict[str, Any]:
        """
        Загружает настройки подключения к OpenSearch из src/config/open_search_settings.yaml.
        
        Returns:
            Словарь с настройками подключения
        """
        try:
            # Читаем настройки из файла (тот же файл, что используется в src)
            settings_path = Path(__file__).parent.parent.parent.parent / "src" / "config" / "open_search_settings.yaml"
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            
            self._opensearch_settings = {
                "hosts": settings.get("hosts", ["localhost:9200"]),
                "login": settings.get("login"),
                "password": settings.get("password", ""),
                "cert_pem_path": settings.get("cert_pem_path"),
                "cert_key_path": settings.get("cert_key_path"),
                "cert_root_path": settings.get("cert_root_path"),
            }

            # Priority override from ES_URL for container/runtime compatibility.
            # Example: ES_URL=http://opensearch-node:9200 -> hosts=["opensearch-node:9200"]
            es_url = os.getenv("ES_URL", "").strip()
            if es_url:
                parsed = urlparse(es_url if "://" in es_url else f"http://{es_url}")
                host = parsed.netloc or parsed.path
                if host:
                    self._opensearch_settings["hosts"] = [host]
            
            logger.debug(f"Loaded OpenSearch settings: hosts={self._opensearch_settings['hosts']}")
            return self._opensearch_settings
            
        except Exception as e:
            logger.error(f"Error loading OpenSearch settings: {e}", exc_info=True)
            # Fallback на настройки по умолчанию
            self._opensearch_settings = {
                "hosts": ["localhost:9200"],
                "login": None,
                "password": "",
                "cert_pem_path": None,
                "cert_key_path": None,
                "cert_root_path": None,
            }
            return self._opensearch_settings
    
    def _setup_opensearch_client(self) -> OpenSearch:
        """
        Настраивает клиент OpenSearch.
        
        Returns:
            Экземпляр OpenSearch клиента
            
        Raises:
            RuntimeError: Если OpenSearch не доступен
        """
        if not OPENSEARCH_AVAILABLE:
            raise RuntimeError("OpenSearch not available. Install with: pip install opensearch-py")
        
        if self._client is not None:
            return self._client
        
        kwargs = {}
        
        # Настройка аутентификации
        if self._opensearch_settings["login"]:
            kwargs["http_auth"] = (
                self._opensearch_settings["login"],
                self._opensearch_settings["password"],
            )
        
        # Настройка SSL сертификатов
        if self._opensearch_settings["cert_pem_path"]:
            kwargs["cert_verify"] = True
            kwargs["client_cert"] = self._opensearch_settings["cert_pem_path"]
            kwargs["client_key"] = self._opensearch_settings["cert_key_path"]
            kwargs["ca_certs"] = self._opensearch_settings["cert_root_path"]
        else:
            kwargs["cert_verify"] = False
            kwargs["use_ssl"] = False

        # Raise request timeout for heavy index create/bulk operations.
        timeout_raw = (
            os.getenv("OPENSEARCH_TIMEOUT", "").strip()
            or os.getenv("ES_TIMEOUT", "").strip()
            or "60"
        )
        if timeout_raw.lower() not in {"", "none", "null", "off", "false", "0"}:
            kwargs["timeout"] = int(timeout_raw)

        kwargs["max_retries"] = int(
            os.getenv("OPENSEARCH_MAX_RETRIES", "").strip()
            or os.getenv("ES_MAX_RETRIES", "").strip()
            or "5"
        )
        retry_on_timeout_raw = (
            os.getenv("OPENSEARCH_RETRY_ON_TIMEOUT", "").strip()
            or os.getenv("ES_RETRY_ON_TIMEOUT", "").strip()
            or "true"
        )
        kwargs["retry_on_timeout"] = retry_on_timeout_raw.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        
        # Создаем клиент
        self._client = OpenSearch(
            hosts=self._opensearch_settings["hosts"],
            **kwargs
        )
        
        logger.debug(f"OpenSearch client initialized for index '{self.index_name}'")
        return self._client
    
    def _create_index(self, force_reload: bool = False) -> None:
        """
        Создает индекс для глав с нужной структурой.
        
        Args:
            force_reload: Если True, удаляет существующий индекс и создает заново
            
        Raises:
            RuntimeError: Если не удается создать индекс
        """
        client = self._setup_opensearch_client()
        
        # Проверяем существование индекса
        index_exists = client.indices.exists(index=self.index_name)
        
        if index_exists and force_reload:
            logger.info(f"Удаление существующего индекса '{self.index_name}'...")
            client.indices.delete(index=self.index_name)
            index_exists = False
        
        if not index_exists:
            logger.info(f"Создание индекса '{self.index_name}'...")
            
            # Определяем структуру индекса
            index_body = {
                "mappings": {
                    "properties": {
                        "source_id": {
                            "type": "integer"
                        },
                        "chapter_text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "created_at": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            try:
                client.indices.create(index=self.index_name, body=index_body)
                logger.info(f"Индекс '{self.index_name}' успешно создан")
            except Exception as e:
                logger.error(f"Ошибка при создании индекса '{self.index_name}': {e}", exc_info=True)
                raise
        else:
            logger.debug(f"Индекс '{self.index_name}' уже существует")
    
    def index_chapters(
        self,
        chapters: List[Document],
        force_reload: bool = False
    ) -> None:
        """
        Сохраняет главы в ElasticSearch индекс.
        
        Args:
            chapters: Список Document объектов с главами (source="book", source_id в metadata)
            force_reload: Удалить и пересоздать индекс если True
            
        Raises:
            ValueError: Если список глав пуст или некорректен
            RuntimeError: Если не удается сохранить главы
        """
        if not chapters:
            logger.warning("Список глав пуст, пропускаем индексацию")
            return
        
        # Валидация глав
        valid_chapters = []
        for chapter in chapters:
            if not isinstance(chapter, Document):
                logger.warning(f"Пропущен невалидный документ: {type(chapter)}")
                continue
            
            source_id = chapter.metadata.get("source_id")
            if source_id is None:
                logger.warning(f"Глава без source_id пропущена: {chapter.metadata}")
                continue
            
            if not chapter.page_content or not chapter.page_content.strip():
                logger.warning(f"Глава {source_id} пуста, пропущена")
                continue
            
            valid_chapters.append(chapter)
        
        if not valid_chapters:
            raise ValueError("Нет валидных глав для индексации")
        
        logger.info(f"Индексация {len(valid_chapters)} глав в индекс '{self.index_name}'...")
        
        # Создаем/обновляем индекс
        self._create_index(force_reload=force_reload)
        
        # Подготавливаем документы для bulk индексации
        client = self._setup_opensearch_client()
        actions = []
        
        for chapter in valid_chapters:
            source_id = chapter.metadata.get("source_id")
            chapter_text = chapter.page_content
            
            action = {
                "_index": self.index_name,
                "_id": str(source_id),  # Используем source_id как document ID
                "_source": {
                    "source_id": int(source_id),
                    "chapter_text": chapter_text,
                    "created_at": datetime.now().isoformat()
                }
            }
            actions.append(action)
        
        # Выполняем bulk индексацию
        try:
            success_count, failed_items = helpers.bulk(
                client, 
                actions, 
                chunk_size=100, 
                raise_on_error=False
            )
            
            failed_count = len(failed_items)
            
            if failed_items:
                logger.warning(f"Ошибки при индексации {failed_count} глав:")
                for item in failed_items[:5]:  # Показываем первые 5 ошибок
                    logger.warning(f"  {item}")
                if failed_count > 5:
                    logger.warning(f"  ... и еще {failed_count - 5} ошибок")
            
            logger.info(
                f"Индексация завершена: {success_count} успешно, {failed_count} ошибок "
                f"из {len(actions)} глав"
            )
            
            if failed_count > 0:
                logger.warning(f"Некоторые главы не были проиндексированы ({failed_count} ошибок)")
            
        except Exception as e:
            logger.error(f"Ошибка при bulk индексации глав: {e}", exc_info=True)
            raise
    
    def get_chapter_by_source_id(self, source_id: int) -> Optional[str]:
        """
        Получает текст главы по source_id.
        
        Args:
            source_id: Номер главы
            
        Returns:
            Текст главы или None если не найдена
            
        Raises:
            RuntimeError: Если OpenSearch не доступен
        """
        if not OPENSEARCH_AVAILABLE:
            raise RuntimeError("OpenSearch not available")
        
        client = self._setup_opensearch_client()
        
        try:
            response = client.get(
                index=self.index_name,
                id=str(source_id)
            )
            
            if response.get("found"):
                return response["_source"].get("chapter_text")
            else:
                logger.debug(f"Глава с source_id={source_id} не найдена в индексе")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при получении главы {source_id}: {e}", exc_info=True)
            return None
    
    def get_chapters_by_source_ids(self, source_ids: List[int]) -> Dict[int, str]:
        """
        Получает тексты нескольких глав по списку source_id.
        
        Args:
            source_ids: Список номеров глав
            
        Returns:
            Словарь {source_id: chapter_text}
        """
        if not OPENSEARCH_AVAILABLE:
            raise RuntimeError("OpenSearch not available")
        
        client = self._setup_opensearch_client()
        
        # Подготавливаем запрос для multi-get
        docs = [{"_index": self.index_name, "_id": str(sid)} for sid in source_ids]
        
        try:
            response = client.mget(body={"docs": docs})
            
            result = {}
            for doc_response in response.get("docs", []):
                if doc_response.get("found"):
                    source_id = int(doc_response["_id"])
                    chapter_text = doc_response["_source"].get("chapter_text")
                    result[source_id] = chapter_text
            
            logger.debug(f"Получено {len(result)} глав из {len(source_ids)} запрошенных")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при получении глав: {e}", exc_info=True)
            return {}
