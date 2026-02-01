from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.preprocess_book.config.preprocess_settings import (
    TEXT_SPLITTER_SEPARATORS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_book(file_path: Path) -> str:
    """
    Загружает книгу из файла.
    
    Args:
        file_path: Путь к файлу с книгой
        
    Returns:
        Текст книги
        
    Raises:
        FileNotFoundError: Если файл не найден
        UnicodeDecodeError: Если не удается декодировать файл
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_book(
    book_text: str,
    separators: List[str] = None,
    split_into_chunks: bool = False,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Разбивает книгу на главы или чанки с метаданными.
    
    По умолчанию разбивает на ГЛАВЫ (большие части по разделителям без ограничения размера).
    Это используется для создания графа знаний и суммаризаций.
    
    Args:
        book_text: Текст книги
        separators: Список разделителей для сплиттера (по умолчанию ["========== ", "***"])
        split_into_chunks: Если True, разбивает на маленькие чанки с ограничением размера.
                          Если False (по умолчанию), разбивает только на главы по разделителям.
        chunk_size: Размер чанка (используется только если split_into_chunks=True)
        chunk_overlap: Перекрытие между чанками (используется только если split_into_chunks=True)
        
    Returns:
        Список Document объектов с метаданными source_id (номер главы или чанка)
    """
    if separators is None:
        separators = TEXT_SPLITTER_SEPARATORS
    
    if split_into_chunks:
        # Разбиение на маленькие чанки с ограничением размера
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        # Разбиение на главы (только по разделителям, без ограничения размера)
        # Это соответствует логике из graphrag.ipynb и summaries.ipynb
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            # НЕ передаем chunk_size и chunk_overlap - разбиваем только по разделителям
        )
    
    texts = text_splitter.create_documents([book_text])
    
    # Добавляем метаданные с source_id
    for num, text in enumerate(texts):
        text.metadata = {"source_id": num}
    
    return texts
