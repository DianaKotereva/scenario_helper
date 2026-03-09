"""
Модуль для разбиения текстов на мелкие чанки по токенам.

Содержит функции для точного подсчета токенов и разбиения
Document объектов на более мелкие чанки с сохранением метаданных.
"""

import logging
from typing import List
import tiktoken
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def calculate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Подсчитывает количество токенов в тексте с помощью tiktoken.
    
    Args:
        text: Текст для подсчета токенов
        encoding_name: Название кодировки (по умолчанию cl100k_base для GPT)
        
    Returns:
        Количество токенов в тексте
    """
    try:
        encoder = tiktoken.get_encoding(encoding_name)
        return len(encoder.encode(text))
    except Exception as e:
        logger.error(f"Ошибка при подсчете токенов: {e}", exc_info=True)
        # Fallback: приблизительная оценка (4 символа на токен)
        return len(text) // 4


def split_text(text: str, chunk_size: int = 512, encoding_name: str = "cl100k_base") -> List[str]:
    """
    Разбивает текст на чанки с учетом ограничений токенов.
    
    Args:
        text: Текст для разбиения
        chunk_size: Максимальный размер чанка в токенах
        encoding_name: Название кодировки
        
    Returns:
        Список текстовых чанков
    """
    if not text:
        return []
    
    try:
        encoder = tiktoken.get_encoding(encoding_name)
        tokens = encoder.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    except Exception as e:
        logger.error(f"Ошибка при разбиении текста: {e}", exc_info=True)
        # Fallback: простое разбиение по символам
        return [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]


def split_documents(
    documents: List[Document], 
    chunk_size: int = 512,
    encoding_name: str = "cl100k_base"
) -> List[Document]:
    """
    Разбивает Document объекты на более мелкие чанки с сохранением метаданных.
    
    Args:
        documents: Список Document объектов для разбиения
        chunk_size: Максимальный размер чанка в токенах
        encoding_name: Название кодировки для подсчета токенов
        
    Returns:
        Список Document объектов с разбитыми текстами
    """
    split_docs = []
    unsplittable_sources = {"nodes", "relations"}
    
    for doc in documents:
        try:
            source = doc.metadata.get("source")
            if source in unsplittable_sources:
                # Keep atomic graph facts unsplitted.
                split_docs.append(doc)
                continue

            # Подсчитываем токены в документе
            doc_tokens = calculate_tokens(doc.page_content, encoding_name)
            
            # Если документ уже меньше размера чанка, добавляем как есть
            if doc_tokens <= chunk_size:
                split_docs.append(doc)
                continue
            
            # Разбиваем текст на чанки
            text_chunks = split_text(doc.page_content, chunk_size, encoding_name)
            
            # Создаем новые Document объекты для каждого чанка
            for chunk_text in text_chunks:
                # Копируем метаданные
                chunk_metadata = doc.metadata.copy()
                
                # Создаем новый Document с разбитым текстом
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                split_docs.append(chunk_doc)
                
        except Exception as e:
            logger.error(
                f"Ошибка при разбиении документа с source_id {doc.metadata.get('source_id', 'unknown')}: {e}",
                exc_info=True
            )
            # В случае ошибки добавляем оригинальный документ
            split_docs.append(doc)
            continue
    
    logger.info(f"Разбито {len(documents)} документов на {len(split_docs)} чанков")
    return split_docs
