"""
Модуль для подготовки документов для загрузки в векторное хранилище.

Содержит класс DocumentPreparer для загрузки и объединения документов
из различных источников: книга, суммаризации, граф знаний.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

from tools.preprocess_book.utils.text_loader import load_book, split_book
from tools.preprocess_book.load_to_vectorstore.graph_converter import book_graph_to_documents
from tools.preprocess_book.load_to_vectorstore.chunk_splitter import split_documents
from src.utils.graph_search import BookGraph

logger = logging.getLogger(__name__)


class DocumentPreparer:
    """
    Класс для подготовки документов из различных источников.
    
    Объединяет документы из книги, суммаризаций и графа знаний
    для последующей загрузки в векторное хранилище.
    """
    
    def __init__(self, chunk_size: int = 512):
        """
        Инициализирует DocumentPreparer.
        
        Args:
            chunk_size: Размер чанка в токенах для разбиения документов
        """
        self.chunk_size = chunk_size
    
    def load_book_chunks(self, book_path: Path) -> List[Document]:
        """
        Загружает книгу и разбивает на ГЛАВЫ (не маленькие чанки).
        
        Разбиение происходит по разделителям ["========== ", "***"] без ограничения размера.
        Это соответствует логике из ноутбуков graphrag.ipynb и put_all_in_elastic.ipynb.
        Маленькие чанки создаются позже в методе split_to_small_chunks.
        
        Args:
            book_path: Путь к файлу с книгой
            
        Returns:
            Список Document объектов с метаданными {"source_id": int, "source": "book"}
            где source_id соответствует номеру главы
        """
        try:
            logger.info(f"Загрузка книги из {book_path}...")
            book_text = load_book(book_path)
            # split_book по умолчанию разбивает на главы (split_into_chunks=False)
            texts = split_book(book_text)
            
            # Добавляем метаданные source для идентификации типа
            for text in texts:
                text.metadata["source"] = "book"
            
            logger.info(f"Загружено {len(texts)} глав книги")
            return texts
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке книги: {e}", exc_info=True)
            raise
    
    def load_summaries(self, summaries_dir: Path) -> List[Document]:
        """
        Загружает суммаризации из директории.
        
        Args:
            summaries_dir: Директория с файлами суммаризаций (*.txt)
            
        Returns:
            Список Document объектов с метаданными {"source_id": int, "source": "summary"}
        """
        summaries = []
        
        if not summaries_dir.exists():
            logger.warning(f"Директория суммаризаций не найдена: {summaries_dir}")
            return summaries
        
        try:
            # Получаем все txt файлы из директории
            summary_files = sorted(
                summaries_dir.glob("*.txt"),
                key=lambda x: int(x.stem) if x.stem.isdigit() else 0
            )
            
            for summary_file in summary_files:
                try:
                    # Извлекаем source_id из имени файла
                    source_id_str = summary_file.stem
                    try:
                        source_id = int(source_id_str)
                    except ValueError:
                        logger.warning(f"Не удалось извлечь source_id из имени файла: {summary_file.name}")
                        continue
                    
                    # Читаем содержимое файла
                    with open(summary_file, "r", encoding="utf-8") as file:
                        summary_text = file.read()
                    
                    # Создаем Document
                    summary_doc = Document(
                        page_content=summary_text,
                        metadata={
                            "source_id": source_id,
                            "source": "summary"
                        }
                    )
                    summaries.append(summary_doc)
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке суммаризации {summary_file}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Загружено {len(summaries)} суммаризаций")
            return summaries
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке суммаризаций: {e}", exc_info=True)
            return summaries
    
    def load_graph_documents(self, book_graph: BookGraph) -> List[Document]:
        """
        Преобразует BookGraph в Document объекты.
        
        Args:
            book_graph: Граф знаний книги
            
        Returns:
            Список Document объектов с узлами и отношениями
        """
        try:
            logger.info("Преобразование графа знаний в Document объекты...")
            documents = book_graph_to_documents(book_graph)
            logger.info(f"Создано {len(documents)} документов из графа")
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка при преобразовании графа: {e}", exc_info=True)
            raise
    
    def prepare_all_documents(
        self,
        book_path: Optional[Path] = None,
        summaries_dir: Optional[Path] = None,
        book_graph: Optional[BookGraph] = None,
        include_book: bool = True,
        include_summaries: bool = True,
        include_graph: bool = True,
    ) -> List[Document]:
        """
        Подготавливает все документы из различных источников.
        
        Args:
            book_path: Путь к файлу с книгой (если include_book=True)
            summaries_dir: Директория с суммаризациями (если include_summaries=True)
            book_graph: Граф знаний (если include_graph=True)
            include_book: Включать ли чанки книги
            include_summaries: Включать ли суммаризации
            include_graph: Включать ли граф знаний
            
        Returns:
            Объединенный список всех Document объектов
        """
        all_documents = []
        
        # Загружаем главы книги (не маленькие чанки - они создаются позже)
        if include_book and book_path:
            try:
                book_docs = self.load_book_chunks(book_path)
                all_documents.extend(book_docs)
            except Exception as e:
                logger.error(f"Не удалось загрузить книгу: {e}", exc_info=True)
        
        # Загружаем суммаризации
        if include_summaries and summaries_dir:
            try:
                summary_docs = self.load_summaries(summaries_dir)
                all_documents.extend(summary_docs)
            except Exception as e:
                logger.error(f"Не удалось загрузить суммаризации: {e}", exc_info=True)
        
        # Загружаем граф знаний
        if include_graph and book_graph:
            try:
                graph_docs = self.load_graph_documents(book_graph)
                all_documents.extend(graph_docs)
            except Exception as e:
                logger.error(f"Не удалось загрузить граф: {e}", exc_info=True)
        
        logger.info(f"Всего подготовлено {len(all_documents)} документов")
        return all_documents
    
    def get_chapters(self, documents: List[Document]) -> List[Document]:
        """
        Извлекает только главы из списка документов.
        
        Главы определяются по метаданным: source="book" и source_id является int.
        Это полные главы, которые НЕ должны разбиваться на маленькие чанки.
        
        Args:
            documents: Список Document объектов
            
        Returns:
            Список Document объектов с главами книги
        """
        chapters = [
            doc for doc in documents
            if doc.metadata.get("source") == "book"
            and isinstance(doc.metadata.get("source_id"), int)
        ]
        logger.debug(f"Извлечено {len(chapters)} глав из {len(documents)} документов")
        return chapters
    
    def split_to_small_chunks(
        self,
        documents: List[Document],
        include_chapters: bool = False,
    ) -> List[Document]:
        """
        Разбивает документы на более мелкие чанки по токенам.
        
        ВАЖНО: Главы (source="book") должны быть исключены из разбиения,
        так как они сохраняются отдельно в обычный ES индекс.
        
        Args:
            documents: Список Document объектов для разбиения (без глав)
            
        Returns:
            Список Document объектов с разбитыми текстами
        """
        # По умолчанию главы исключаются из разбиения (историческое поведение),
        # но при include_chapters=True разбиваем всё, кроме атомарных граф-документов.
        if include_chapters:
            documents_to_split = list(documents)
        else:
            documents_to_split = [
                doc for doc in documents
                if not (doc.metadata.get("source") == "book" and isinstance(doc.metadata.get("source_id"), int))
            ]
        
        logger.info(f"Разбиение {len(documents_to_split)} документов на чанки по {self.chunk_size} токенов...")
        split_docs = split_documents(documents_to_split, chunk_size=self.chunk_size)
        logger.info(f"После разбиения получено {len(split_docs)} чанков")
        return split_docs
