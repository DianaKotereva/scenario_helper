"""
Конфигурация для модуля предобработки книги.
"""

from tools.preprocess_book.config.preprocess_settings import (
    BASE_DIR,
    BOOK_INPUT_DIR,
    RESULTS_DIR,
    GRAPH_NODES_DIR,
    GRAPH_RELATIONS_DIR,
    OUTPUT_DIR,
    LLM_TYPE,
    TEXT_SPLITTER_SEPARATORS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VERIFICATION_LAST_N,
)

__all__ = [
    "BASE_DIR",
    "BOOK_INPUT_DIR",
    "RESULTS_DIR",
    "GRAPH_NODES_DIR",
    "GRAPH_RELATIONS_DIR",
    "OUTPUT_DIR",
    "LLM_TYPE",
    "TEXT_SPLITTER_SEPARATORS",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "VERIFICATION_LAST_N",
]
