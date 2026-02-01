#!/usr/bin/env python3
"""
Единая точка входа для процесса раскладки книги в граф знаний.

Использование:
    python -m tools.preprocess_book.main --book-path data/Ten-i-Plama.txt --output-path data/bookgraph.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from src.utils.graph_search import BookGraph
from tools.preprocess_book.config.preprocess_settings import (
    OUTPUT_DIR,
    SUMMARIES_DIR,
    VECTORSTORE_CHUNK_SIZE,
    VECTORSTORE_PICKLE_PATH,
    PARALLEL_CONCURRENCY,
    GRAPH_BUILD_CONCURRENCY,
)
from tools.preprocess_book.load_to_vectorstore import (
    ChapterIndexer,
    DocumentPreparer,
    VectorStoreLoader,
)
from tools.preprocess_book.make_graph.extraction_service import ExtractionService
from tools.preprocess_book.make_graph.graph_builder import GraphBuilder
from tools.preprocess_book.make_graph.node_processor import NodeProcessor
from tools.preprocess_book.make_graph.relation_processor import RelationProcessor
from tools.preprocess_book.make_graph.verification_service import VerificationService
from tools.preprocess_book.make_summaries import (
    SummarizationPrompt,
    SummarizationService,
)
from tools.preprocess_book.prompts.extract_names import ExtractNames
from tools.preprocess_book.prompts.verificator import Verification
from tools.preprocess_book.storage.storage import FileManager
from tools.preprocess_book.utils.llm_factory import create_llm
from tools.preprocess_book.utils.text_loader import load_book, split_book

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocess_book.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция для обработки книги и построения графа."""
    parser = argparse.ArgumentParser(
        description="Обработка книги и построение графа знаний"
    )
    parser.add_argument(
        "--book-path", type=str, required=True, help="Путь к файлу с книгой"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Путь для сохранения финального графа (по умолчанию data/bookgraph.pkl)",
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        default=None,
        help="Тип LLM (deepseek или chatgpt). По умолчанию из настроек",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Пропустить этап экстракции и использовать существующие результаты",
    )
    parser.add_argument(
        "--create-summaries",
        action="store_true",
        help="Создать суммаризации глав книги",
    )
    parser.add_argument(
        "--summaries-output-dir",
        type=str,
        default=None,
        help="Директория для сохранения суммаризаций (по умолчанию tools/preprocess_book/summaries)",
    )
    parser.add_argument(
        "--load-to-vectorstore",
        action="store_true",
        help="Загрузить документы (книга, суммаризации, граф) в векторное хранилище",
    )
    parser.add_argument(
        "--vectorstore-pickle-path",
        type=str,
        default=None,
        help="Путь к pickle файлу для сохранения документов (по умолчанию data/all_langchain_chunks.pkl)",
    )
    parser.add_argument(
        "--vectorstore-force-reload",
        action="store_true",
        help="Принудительная перезагрузка индекса векторного хранилища",
    )

    args = parser.parse_args()

    book_path = Path(args.book_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = OUTPUT_DIR / "bookgraph.pkl"

    try:
        # Инициализация LLM
        logger.info("Инициализация LLM...")
        llm = create_llm(llm_type=args.llm_type)

        # Создание промптов
        logger.info("Создание промптов...")
        json_parser = JsonOutputParser()
        extractor = ExtractNames(llm=llm, parser=json_parser)
        verificator = Verification(llm=llm, parser=json_parser)

        # Инициализация сервисов
        logger.info("Инициализация сервисов...")
        verification_service = VerificationService(verificator)
        node_processor = NodeProcessor(verification_service)
        relation_processor = RelationProcessor()

        extraction_service = ExtractionService(extractor)
        graph_builder = GraphBuilder(node_processor, relation_processor)

        # Этап 1: Загрузка и разбиение книги на ГЛАВЫ
        # Важно: разбиение происходит на главы (большие части по разделителям),
        # а не на маленькие чанки. Маленькие чанки создаются только для vectorstore.
        texts = None
        if not args.skip_extraction:
            logger.info("Этап 1: Загрузка и разбиение книги на главы...")
            book_text = load_book(book_path)
            # split_book по умолчанию разбивает на главы (split_into_chunks=False)
            texts = split_book(book_text)
            logger.info(f"Книга разбита на {len(texts)} глав")

            # Этап 1.5: Создание суммаризаций (если запрошено)
            if args.create_summaries:
                logger.info("Этап 1.5: Создание суммаризаций глав...")
                summaries_output_dir = (
                    Path(args.summaries_output_dir)
                    if args.summaries_output_dir
                    else SUMMARIES_DIR
                )
                summaries_output_dir.mkdir(parents=True, exist_ok=True)

                summarization_prompt = SummarizationPrompt(llm=llm)
                summarization_service = SummarizationService(summarization_prompt)
                summaries = summarization_service.create_summaries(
                    texts, summaries_output_dir, concurrency=PARALLEL_CONCURRENCY
                )

                logger.info(f"Создано {len([s for s in summaries if s])} суммаризаций")
                logger.info(f"Суммаризации сохранены в: {summaries_output_dir}")

            # Этап 2: Экстракция данных из глав книги
            logger.info("Этап 2: Экстракция сущностей и отношений из глав книги...")
            all_summarizations = extraction_service.extract_from_texts(
                texts, concurrency=PARALLEL_CONCURRENCY
            )
            logger.info(f"Обработано {len(all_summarizations)} глав")
        else:
            logger.info(
                "Пропуск этапа экстракции (используются существующие результаты)"
            )

        # Этап 3: Построение графа из результатов экстракции
        logger.info("Этап 3: Построение графа...")
        all_book_nodes, relation_graphs = graph_builder.build_graph_from_results(
            concurrency=GRAPH_BUILD_CONCURRENCY
        )

        logger.info(
            f"Построен граф с {len(all_book_nodes.nodes)} узлами и "
            f"{len(relation_graphs.relationships)} отношениями"
        )

        # Создание финального графа
        book_graph = BookGraph(nodes=all_book_nodes, relationships=relation_graphs)

        # Сохранение финального графа
        logger.info(f"Сохранение графа в {output_path}...")
        FileManager.save_pickle(book_graph, output_path)

        logger.info("Обработка завершена успешно!")
        logger.info(f"Результаты сохранены в: {output_path}")

        # Этап 4: Загрузка в векторное хранилище (если запрошено)
        if args.load_to_vectorstore:
            logger.info("Этап 4: Загрузка документов в векторное хранилище...")

            try:
                # Определяем пути
                vectorstore_pickle_path = (
                    Path(args.vectorstore_pickle_path)
                    if args.vectorstore_pickle_path
                    else VECTORSTORE_PICKLE_PATH
                )
                summaries_dir = (
                    Path(args.summaries_output_dir)
                    if args.summaries_output_dir
                    else SUMMARIES_DIR
                )

                # Создаем DocumentPreparer
                preparer = DocumentPreparer(chunk_size=VECTORSTORE_CHUNK_SIZE)

                # Подготавливаем все документы
                logger.info("Подготовка документов для загрузки...")
                all_documents = preparer.prepare_all_documents(
                    book_path=book_path,
                    summaries_dir=summaries_dir if summaries_dir.exists() else None,
                    book_graph=book_graph,
                    include_book=True,
                    include_summaries=summaries_dir.exists(),
                    include_graph=True,
                )

                # Отделяем главы от остальных документов
                logger.info("Разделение глав и остальных документов...")
                chapters = preparer.get_chapters(all_documents)
                other_documents = [doc for doc in all_documents if doc not in chapters]
                logger.info(
                    f"Найдено {len(chapters)} глав и {len(other_documents)} других документов"
                )

                # Разбиваем на мелкие чанки только остальные документы (не главы)
                logger.info("Разбиение документов (кроме глав) на мелкие чанки...")
                split_docs = preparer.split_to_small_chunks(other_documents)

                # Загружаем чанки в векторное хранилище
                logger.info("Загрузка чанков в векторное хранилище...")
                loader = VectorStoreLoader()
                loader.load_documents(
                    documents=split_docs,
                    output_pickle_path=vectorstore_pickle_path,
                    force_reload=args.vectorstore_force_reload,
                )

                logger.info("Чанки успешно загружены в векторное хранилище!")
                logger.info(f"Pickle файл сохранен в: {vectorstore_pickle_path}")

                # Сохраняем главы в отдельный обычный ES индекс (не векторный)
                if chapters:
                    logger.info(
                        f"Сохранение {len(chapters)} глав в отдельный ES индекс..."
                    )
                    chapter_indexer = ChapterIndexer()
                    chapter_indexer.index_chapters(
                        chapters=chapters,
                        force_reload=args.vectorstore_force_reload,
                    )
                    logger.info("Главы успешно сохранены в ES индекс!")
                else:
                    logger.warning(
                        "Главы не найдены, пропускаем сохранение в ES индекс"
                    )

            except Exception as e:
                logger.error(
                    f"Ошибка при загрузке в векторное хранилище: {e}", exc_info=True
                )
                # Не прерываем выполнение, только логируем ошибку

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
