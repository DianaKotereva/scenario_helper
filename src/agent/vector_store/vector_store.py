import logging
import hashlib
import time
from urllib.parse import urlparse
from itertools import islice
from pathlib import Path
from typing import List, Optional

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.config import settings

try:
    from langchain_community.vectorstores import FAISS

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from langchain_community.vectorstores import OpenSearchVectorSearch
    from opensearchpy import helpers

    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logging.warning("OpenSearch not available. Install with: pip install opensearch-py")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStore:
    """A unified interface for managing vector stores with document processing."""

    def __init__(
        self,
        embedding_function: Embeddings,
        index_name: str,
        batch_size: int = 100,
        pickle_documents_path: Optional[str] = "./data/langhcain_docs.pkl",
        passage_prefix: Optional[str] = "passage: ",
        full_reload: bool = False,
        delete_old: bool = False,
        faiss_path: Optional[str] = "./faiss_index",  # Путь для сохранения FAISS
        setup_index: bool = True,
        changed_source_ids: Optional[List[int]] = None,
    ):
        if not embedding_function:
            raise ValueError("Embedding function is required")

        self.embedding_function = embedding_function
        self.batch_size = batch_size
        self.pickle_documents_path = Path(pickle_documents_path)
        self.index_name = index_name
        self._passage_prefix = passage_prefix
        self.full_reload = full_reload
        self.delete_old = delete_old
        self.faiss_path = Path(faiss_path) if faiss_path else None
        self.setup_index = setup_index
        self.changed_source_ids = changed_source_ids or []
        self.store = self._initialize_store()
        if self.setup_index:
            self._setup_index()

    def _initialize_store(self):
        """Initialize the vector store client based on settings."""
        vector_store_type = getattr(settings, "USE_FAISS", False)

        if vector_store_type:
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS is not available. Please install faiss-cpu")
            return self._initialize_faiss()
        else:
            if not OPENSEARCH_AVAILABLE:
                raise ImportError(
                    "OpenSearch is not available. Please install opensearch-py"
                )
            return self._initialize_opensearch()

    def _initialize_faiss(self):
        """Initialize FAISS vector store."""
        # Создаем директорию для FAISS если не существует
        if self.faiss_path:
            self.faiss_path.mkdir(parents=True, exist_ok=True)

        faiss_index_path = self.faiss_path / f"{self.index_name}.faiss"

        # Если индекс существует и не требуется полная перезагрузка, загружаем его
        if faiss_index_path.exists() and not self.full_reload:
            logger.info(f"Loading existing FAISS index from {faiss_index_path}")
            return FAISS.load_local(
                str(self.faiss_path),
                self.embedding_function,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("Creating new FAISS index")
            return None

    # TODO: сделать open_search_settings

    def _initialize_opensearch(self) -> OpenSearchVectorSearch:
        """Initialize the OpenSearch vector store client."""
        cfg = settings.open_search_settings or {}
        if not isinstance(cfg, dict):
            cfg = {}

        kwargs = {}
        if cfg.get("login") and cfg.get("password"):
            kwargs["http_auth"] = (
                cfg.get("login"),
                cfg.get("password"),
            )
        if cfg.get("cert_pem_path") is not None:
            kwargs["cert_verify"] = True
            kwargs["client_cert"] = cfg.get("cert_pem_path")
            kwargs["client_key"] = cfg.get("cert_key_path")
            kwargs["ca_certs"] = cfg.get("cert_root_path")
        else:
            kwargs["cert_verify"] = False
            kwargs["use_ssl"] = False

        opensearch_url = settings.ES_URL
        if not opensearch_url.startswith(("http://", "https://")):
            opensearch_url = f"http://{opensearch_url}"
        parsed = urlparse(opensearch_url)
        opensearch_host = opensearch_url if parsed.netloc else f"http://{parsed.path}"

        # Connection behavior for long-running scan/incremental operations.
        # ES_TIMEOUT=None means "no explicit client timeout".
        if getattr(settings, "ES_TIMEOUT", None) is not None:
            kwargs["timeout"] = settings.ES_TIMEOUT
        kwargs["max_retries"] = getattr(settings, "ES_MAX_RETRIES", 5)
        kwargs["retry_on_timeout"] = getattr(settings, "ES_RETRY_ON_TIMEOUT", True)

        return OpenSearchVectorSearch(
            opensearch_url=opensearch_host,
            index_name=self.index_name,
            embedding_function=self.embedding_function,
            engine="faiss",
            **kwargs,
        )

    def add_docs(self, to_add):
        """Add documents to the vector store."""
        if to_add:
            doc_iter = iter(to_add)
            logger.info(f"Add {len(to_add)} new documents")
            while True:
                batch = list(islice(doc_iter, int(self.batch_size)))
                if not batch:
                    break
                self.store.add_documents(batch, batch_size=self.batch_size)
            logger.info(f"Added {len(to_add)} new documents")

    @staticmethod
    def _stable_doc_id(doc: Document) -> int:
        meta = doc.metadata or {}
        source = str(meta.get("source", "unknown"))
        source_id = str(meta.get("source_id", "unknown"))
        uid = str(meta.get("doc_uid", ""))
        payload = f"{source}|{source_id}|{uid}|{doc.page_content}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:15]
        return int(digest, 16)

    def _doc_identity(self, doc: Document) -> str:
        meta = doc.metadata
        if "source_id" not in meta:
            raise ValueError("You should provide source_id")
        if "id" not in meta:
            meta["id"] = self._stable_doc_id(doc)
        return f"{meta['source_id']}_{meta['id']}"

    def _embed_with_retry(self, texts: List[str], retries: int = 3, backoff: float = 1.5):
        attempt = 0
        while True:
            try:
                return self.store.embedding_function.embed_documents(texts)
            except Exception:
                attempt += 1
                if attempt > retries:
                    raise
                sleep_s = backoff ** attempt
                logger.warning(
                    "Embedding batch failed (attempt %s/%s), retry in %.1fs",
                    attempt,
                    retries,
                    sleep_s,
                )
                time.sleep(sleep_s)

    def _upsert_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        client = self.store.client
        batch_size = max(1, int(self.batch_size))
        total = 0
        started = time.time()
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.page_content for d in batch]
            vectors = self._embed_with_retry(texts)
            actions = []
            for doc, vector in zip(batch, vectors):
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        "_id": self._doc_identity(doc),
                        "_source": {
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "vector_field": vector,
                        },
                    }
                )
            helpers.bulk(client, actions)
            total += len(actions)
        logger.info(
            "Upserted %s docs into '%s' in %.2fs",
            total,
            self.index_name,
            time.time() - started,
        )

    def _delete_by_source_ids(self, source_ids: List[int]) -> None:
        if not source_ids:
            return
        client = self.store.client
        query = {"query": {"terms": {"metadata.source_id": source_ids}}}
        client.delete_by_query(
            index=self.index_name,
            body=query,
            conflicts="proceed",
            refresh=True,
        )
        logger.info("Deleted existing docs by source_ids: %s", len(source_ids))

    def _incremental_change(self, new_docs) -> None:
        """Handle incremental changes for OpenSearch without full-index scan."""
        if not OPENSEARCH_AVAILABLE:
            raise RuntimeError("OpenSearch not available")

        if self.changed_source_ids:
            self._delete_by_source_ids(self.changed_source_ids)
        elif self.delete_old:
            logger.warning("delete_old=True without changed_source_ids may keep stale docs")

        self._upsert_documents(new_docs)
        logger.info("Incremental update completed for index '%s'", self.index_name)

    def batch_documents_by_tokens(
        self, documents, max_tokens_per_batch, encoding_name="cl100k_base"
    ):
        encoding = tiktoken.get_encoding(encoding_name)
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in documents:
            # Подсчитываем токены в документе
            doc_tokens = len(encoding.encode(doc.page_content))
            # Если добавление этого документа превысит лимит, то завершаем текущий батч
            if current_tokens + doc_tokens > max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _setup_index(self) -> None:
        """Configure the index and load documents if needed."""
        vector_store_type = getattr(settings, "USE_FAISS", False)

        documents = self._load_documents_from_pickle()

        if vector_store_type:
            # Для FAISS просто добавляем документы если они есть
            if documents and (self.store is None or self.full_reload):
                batches = self.batch_documents_by_tokens(
                    documents, max_tokens_per_batch=self.batch_size * 512
                )
                if not batches:
                    raise ValueError("No documents to index.")
                self.store = FAISS.from_documents(batches[0], self.embedding_function)
                for batch in batches[1:]:
                    self.store.add_documents(batch)
                if self.faiss_path:
                    self.store.save_local(
                        str(self.faiss_path), index_name=self.index_name
                    )

        else:
            # OpenSearch логика
            index_exists = self.store.index_exists(index_name=self.index_name)
            if self.full_reload:
                if index_exists:
                    self.store.delete_index(index_name=self.index_name)
                    index_exists = False

            if index_exists:
                self._incremental_change(documents)
            else:
                embed_dim = len(self.embedding_function.embed_query("hello"))
                self.store.create_index(
                    dimension=embed_dim, index_name=self.index_name
                )
                if documents:
                    self._upsert_documents(documents)

    def _load_documents_from_pickle(self) -> List[Document]:
        """Load documents from a pickle file."""
        import pickle

        if not self.pickle_documents_path.exists():
            logger.warning(f"Pickle file {self.pickle_documents_path} not found")
            return []

        try:
            with open(self.pickle_documents_path, "rb") as f:
                documents = pickle.load(f)
        except Exception as e:
            logger.warning(
                "Failed to load pickle %s, using empty document list. Error: %s",
                self.pickle_documents_path,
                e,
            )
            return []
        if self._passage_prefix:
            documents = [
                Document(
                    page_content=self._passage_prefix + doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in documents
            ]

        for i in documents:
            if "source_id" not in i.metadata:
                raise ValueError("You should provide source_id")
            if "id" not in i.metadata:
                i.metadata["id"] = self._stable_doc_id(i)
        return documents

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        """Search for similar documents."""
        return self.store.similarity_search(query, k=k, **kwargs)

    def as_retriever(self, **kwargs):
        """Get the vector store as a retriever."""
        return self.store.as_retriever(**kwargs)


if __name__ == "__main__":
    from langchain_core.embeddings import FakeEmbeddings

    # Тестирование с обоими типами хранилищ
    for store_type in ["opensearch", "faiss"]:
        print(f"Testing {store_type}...")

        # Временно устанавливаем тип хранилища
        settings.vector_store_type = store_type

        embedding = FakeEmbeddings()

        store = VectorStore(
            embedding=embedding,
            index_name=f"test_index_{store_type}",
            faiss_path="./test_faiss_index",
        )

        # Тест поиска
        results = store.similarity_search("test query", k=2)
        print(f"{store_type} results: {len(results)}")
