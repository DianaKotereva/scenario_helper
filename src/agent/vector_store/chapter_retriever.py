"""
Модуль для получения глав книги из обычного ElasticSearch индекса.

Содержит класс ChapterRetriever для получения полных текстов глав
из отдельного индекса (не векторного) по source_id.
"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from opensearchpy import OpenSearch
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("OpenSearch not available. Install with: pip install opensearch-py")

from src.config import settings

logger = logging.getLogger(__name__)


class ChapterRetriever:
    """
    Класс для получения глав книги из обычного ElasticSearch индекса.
    
    Используется для добавления полных текстов глав в контекст агента
    на основе source_id, найденных в результатах поиска.
    """
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Инициализирует ChapterRetriever.
        
        Args:
            index_name: Имя индекса (по умолчанию из настроек)
        """
        self.index_name = index_name or settings.CHAPTERS_INDEX_NAME
        self._client = None
        self._load_opensearch_settings()
    
    def _load_opensearch_settings(self) -> Dict[str, Any]:
        """
        Загружает настройки подключения к OpenSearch из src/config/open_search_settings.yaml.
        
        Returns:
            Словарь с настройками подключения
        """
        try:
            # Читаем настройки из файла
            settings_path = Path(__file__).parent.parent.parent / "config" / "open_search_settings.yaml"
            with open(settings_path, "r", encoding="utf-8") as f:
                opensearch_config = yaml.safe_load(f)
            
            self._opensearch_settings = {
                "hosts": opensearch_config.get("hosts", ["localhost:9200"]),
                "login": opensearch_config.get("login"),
                "password": opensearch_config.get("password", ""),
                "cert_pem_path": opensearch_config.get("cert_pem_path"),
                "cert_key_path": opensearch_config.get("cert_key_path"),
                "cert_root_path": opensearch_config.get("cert_root_path"),
            }
            
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
        
        # Создаем клиент
        self._client = OpenSearch(
            hosts=self._opensearch_settings["hosts"],
            **kwargs
        )
        
        logger.debug(f"OpenSearch client initialized for index '{self.index_name}'")
        return self._client
    
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
        
        try:
            client = self._setup_opensearch_client()
            
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
            logger.warning(f"Ошибка при получении главы {source_id}: {e}", exc_info=True)
            return None
    
    def get_chapters_by_source_ids(
        self, 
        source_ids: List[int],
        max_chapters: Optional[int] = None
    ) -> Dict[int, str]:
        """
        Получает тексты нескольких глав по списку source_id.
        
        Использует bulk-запрос (mget) для эффективного получения нескольких глав.
        
        Args:
            source_ids: Список номеров глав
            max_chapters: Максимальное количество глав для возврата (если None - все)
            
        Returns:
            Словарь {source_id: chapter_text}
        """
        if not OPENSEARCH_AVAILABLE:
            logger.warning("OpenSearch not available, returning empty dict")
            return {}
        
        if not source_ids:
            return {}
        
        # Ограничиваем количество глав если указано
        if max_chapters is not None:
            source_ids = source_ids[:max_chapters]
        
        try:
            client = self._setup_opensearch_client()
            
            # Подготавливаем запрос для multi-get
            docs = [{"_index": self.index_name, "_id": str(sid)} for sid in source_ids]
            
            response = client.mget(body={"docs": docs})
            
            result = {}
            for doc_response in response.get("docs", []):
                if doc_response.get("found"):
                    source_id = int(doc_response["_id"])
                    chapter_text = doc_response["_source"].get("chapter_text")
                    if chapter_text:
                        result[source_id] = chapter_text
            
            logger.debug(f"Получено {len(result)} глав из {len(source_ids)} запрошенных")
            return result
            
        except Exception as e:
            logger.warning(f"Ошибка при получении глав: {e}", exc_info=True)
            return {}
