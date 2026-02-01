"""
Модуль для загрузки данных в векторное хранилище.

Содержит классы для подготовки документов (книга, суммаризации, граф знаний)
и загрузки их в векторное хранилище (OpenSearch/FAISS).
"""

from .document_preparer import DocumentPreparer
from .vectorstore_loader import VectorStoreLoader
from .chapter_indexer import ChapterIndexer
from .graph_converter import book_graph_to_documents
from .chunk_splitter import split_documents, calculate_tokens

__all__ = [
    "DocumentPreparer",
    "VectorStoreLoader",
    "ChapterIndexer",
    "book_graph_to_documents",
    "split_documents",
    "calculate_tokens",
]
