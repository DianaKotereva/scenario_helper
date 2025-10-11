import os
import pickle
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from src.agent.vector_store.vector_store import VectorStore
from src.config import settings as settings


class OpenSearchStore:
    def __init__(self, documents: List[Document] = None, force_reload: bool = False):
        self.force_reload = force_reload
        self._embeddings = self._setup_embeddings()
        self._vector_store = self._setup_vector_store()
        self._retriever = self._setup_retriever(documents)

    def _setup_embeddings(
        self,
    ) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMB_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        return embeddings

    def _setup_retriever(self, documents: List[Document] = None) -> EnsembleRetriever:
        pickle_documents_path = settings.ES_PICKLE_DOCUMENTS_PATH
        if not documents:
            if pickle_documents_path and os.path.exists(pickle_documents_path):
                with open(pickle_documents_path, "rb") as f:
                    documents = pickle.load(f)
            else:
                raise FileNotFoundError(
                    "Pickle file with documents not found. Please check the path."
                )
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=settings.BM25_K,
        )

        es_retriever = self._vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.ES_RETRIEVER_K,
                "score_threshold": settings.ES_SIMILARITY_THRESHOLD,
            },
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[es_retriever, bm25_retriever],
            weights=[0.8, 0.2],
        )
        return ensemble_retriever

    def _setup_vector_store(self):
        vector_store = VectorStore(
            embedding_function=self._embeddings,
            index_name=settings.ES_INDEX_NAME,
            batch_size=settings.ES_BATCH_SIZE,
            pickle_documents_path=settings.ES_PICKLE_DOCUMENTS_PATH,
            full_reload=self.force_reload,
            faiss_path=settings.FAISS_PATH,
        )
        return vector_store.store


elastic_store = OpenSearchStore(force_reload=False)
retriever = elastic_store._retriever
