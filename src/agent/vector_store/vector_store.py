import logging
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
        self.store = self._initialize_store()
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
        kwargs = {}
        if settings.open_search_settings.login is not None:
            kwargs["http_auth"] = (
                settings.open_search_settings.login,
                settings.open_search_settings.password,
            )
        if settings.open_search_settings.cert_pem_path is not None:
            kwargs["cert_verify"] = True
            kwargs["client_cert"] = settings.open_search_settings.cert_pem_path
            kwargs["client_key"] = settings.open_search_settings.cert_key_path
            kwargs["ca_certs"] = settings.open_search_settings.cert_root_path
        else:
            kwargs["cert_verify"] = False
            kwargs["use_ssl"] = False

        return OpenSearchVectorSearch(
            opensearch_url=settings.open_search_settings.hosts[0],
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
            # self.store.add_documents(
            #     to_add, batch_size=self.batch_size, bulk_size=len(to_add)
            # )
            logger.info(f"Added {len(to_add)} new documents")

    def _incremental_change(self, new_docs) -> None:
        """Handle incremental changes for OpenSearch."""
        if not OPENSEARCH_AVAILABLE:
            raise RuntimeError("OpenSearch not available")

        client = self.store.client
        existing_docs = {}
        query = {"query": {"match_all": {}}, "_source": ["text", "metadata"]}

        to_add = []
        to_update = []
        to_remove = []

        for hit in helpers.scan(client, index=self.index_name, query=query):
            try:
                meta = hit["_source"]["metadata"]
                if "source_id" in meta:
                    doc_id = str(meta["source_id"]) + "_" + str(meta["id"])
                else:
                    raise ValueError("You should provide source_id")
                existing_docs[doc_id] = {
                    "es_id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"]["metadata"],
                }
            except KeyError:
                to_remove.append(hit["_id"])

        new_docs_dict = {}
        for doc in new_docs:
            meta = doc.metadata
            if "source_id" in meta:
                doc_id = str(meta["source_id"]) + "_" + str(meta["id"])
            else:
                raise ValueError("You should provide source_id")
            new_docs_dict[doc_id] = doc

        # Поиск новых и измененных документов
        for doc_id, new_doc in new_docs_dict.items():
            if doc_id not in existing_docs:
                to_add.append(new_doc)
            else:
                existing_doc = existing_docs[doc_id]
                if (
                    new_doc.page_content != existing_doc["text"]
                    or new_doc.metadata != existing_doc["metadata"]
                ):
                    # Сохраняем es_id для обновления
                    new_doc.metadata["_es_id"] = existing_doc["es_id"]
                    to_update.append(new_doc)

        # Поиск документов для удаления
        if self.delete_old:
            existing_ids = set(existing_docs.keys())
            new_ids = set(new_docs_dict.keys())
            for doc_id in existing_ids - new_ids:
                to_remove.append(existing_docs[doc_id]["es_id"])

        # Выполнение операций
        logger.info(
            f"Update stats: +{len(to_add)} to add, ±{len(to_update)} to update, -{len(to_remove)} to delete"
        )

        # Добавление новых документов
        if to_add:
            self.add_docs(to_add)

        # Обновление измененных документов
        if to_update:
            bulk_actions = []
            for doc in to_update:
                es_id = doc.metadata.pop("_es_id")
                doc_body = {"content": doc.page_content, "metadata": doc.metadata}

                vector = self.store.embedding_function.embed_documents(
                    [doc.page_content]
                )[0]
                doc_body["vector"] = vector

                bulk_actions.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        "_id": es_id,
                        "_source": doc_body,
                    }
                )

            helpers.bulk(client, bulk_actions)
            logger.info(f"Updated {len(to_update)} documents")

        # Удаление лишних документов
        if to_remove:
            bulk_actions = [
                {"_op_type": "delete", "_index": self.index_name, "_id": es_id}
                for es_id in to_remove
            ]
            helpers.bulk(client, bulk_actions)
            logger.info(f"Deleted {len(to_remove)} documents")

        logger.info(f"Incremental update completed for index '{self.index_name}'")

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
                if documents:
                    self.add_docs(documents)
                else:
                    embed_dim = len(self.embedding_function.embed_query("hello"))
                    self.store.create_index(
                        dimension=embed_dim, index_name=self.index_name
                    )

    def _load_documents_from_pickle(self) -> List[Document]:
        """Load documents from a pickle file."""
        import pickle

        if not self.pickle_documents_path.exists():
            logger.warning(f"Pickle file {self.pickle_documents_path} not found")
            return []

        with open(self.pickle_documents_path, "rb") as f:
            documents = pickle.load(f)
        if self._passage_prefix:
            documents = [
                Document(
                    page_content=self._passage_prefix + doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in documents
            ]

        all_graph_nodes = set()

        for i in documents:
            if "source_id" in i.metadata:
                graph_node = str(i.metadata["source_id"])
            else:
                raise ValueError("You should provide source_id")
            if graph_node not in all_graph_nodes:
                n = 0
            else:
                n += 1
            i.metadata["id"] = n
            all_graph_nodes.add(graph_node)
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
