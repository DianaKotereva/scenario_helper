#!/usr/bin/env python3
"""
Единая точка входа для процесса раскладки книги в граф знаний.

Использование:
    python -m tools.preprocess_book.main --book-path data/Ten-i-Plama.txt --output-path data/bookgraph.pkl
"""

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from src.utils.graph_search import BookGraph
from tools.preprocess_book.config.preprocess_settings import (
    EXTRACTION_CHAPTER_BATCH_SIZE,
    OUTPUT_DIR,
    SUMMARIES_DIR,
    VECTORSTORE_CHUNK_SIZE,
    VECTORSTORE_PICKLE_PATH,
    PARALLEL_CONCURRENCY,
    RESULTS_DIR,
    GRAPH_NODES_DIR,
    GRAPH_RELATIONS_DIR,
)
from tools.preprocess_book.load_to_vectorstore import (
    ChapterIndexer,
    DocumentPreparer,
    VectorStoreLoader,
)
from tools.preprocess_book.load_to_vectorstore.quality_gates import (
    expected_graph_fact_counts,
    validate_chapters,
    validate_graph_documents,
    validate_non_chapter_documents,
)
from tools.preprocess_book.make_graph.extraction_service import ExtractionService
from tools.preprocess_book.make_graph.batch_graph_builder import BatchGraphBuilder
from tools.preprocess_book.make_graph.node_processor import NodeProcessor
from tools.preprocess_book.make_graph.relation_processor import RelationProcessor
from tools.preprocess_book.make_graph.verification_service import VerificationService
from tools.preprocess_book.make_summaries import (
    SummarizationPrompt,
    SummarizationService,
)
from tools.preprocess_book.make_graph.v2_quality_gates import evaluate_merge_quality
from tools.preprocess_book.make_graph.v2_verification_pipeline import VerificationPipelineV2
from tools.preprocess_book.prompts.extract_names import ExtractNames
from tools.preprocess_book.prompts.extract_relations import ExtractRelations
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


def _source_hash_map(documents):
    by_source = {}
    for doc in documents:
        source_id = doc.metadata.get("source_id")
        if not isinstance(source_id, int):
            continue
        uid = str(doc.metadata.get("doc_uid", ""))
        row = f"{uid}|{doc.page_content}"
        by_source.setdefault(source_id, []).append(row)

    res = {}
    for source_id, rows in by_source.items():
        payload = "\n".join(sorted(rows))
        res[source_id] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return res


def _load_manifest(path: Path):
    if not path.exists():
        return {"source_hashes": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"source_hashes": {}}


def _save_manifest(path: Path, source_hashes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"source_hashes": source_hashes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _detect_changed_source_ids(current_hashes, previous_hashes):
    current_keys = set(current_hashes.keys())
    previous_keys = set(previous_hashes.keys())
    changed = {
        sid
        for sid in current_keys
        if previous_hashes.get(str(sid)) != current_hashes.get(sid)
    }
    removed = {int(sid) for sid in previous_keys - {str(i) for i in current_keys}}
    return sorted(changed | removed)


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
        dest="load_to_vectorstore",
        action="store_true",
        default=True,
        help="Load documents (book, summaries, graph) into vectorstore and chapters index (enabled by default)",
    )
    parser.add_argument(
        "--skip-vectorstore",
        dest="load_to_vectorstore",
        action="store_false",
        help="Skip loading documents into vectorstore and chapters index",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Быстрый отладочный режим (не более 5 глав)",
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=0,
        help="Ограничить количество обрабатываемых глав (0 = все)",
    )
    parser.add_argument(
        "--export-snapshot",
        action="store_true",
        help="Экспортировать snapshot jsonl (chapters/chunks/entities/relations/index)",
    )
    parser.add_argument(
        "--snapshot-output-dir",
        type=str,
        default=None,
        help="Директория для snapshot-файлов (по умолчанию tools/preprocess_book/_snapshot)",
    )
    parser.add_argument(
        "--snapshot-max-files",
        type=int,
        default=0,
        help="Ограничение числа файлов для snapshot (0 = все обработанные главы)",
    )
    parser.add_argument(
        "--snapshot-validate",
        action="store_true",
        help="Запустить валидацию snapshot и сохранить validation_report.json",
    )

    parser.add_argument(
        "--merge-engine",
        type=str,
        default="v2like",
        choices=["classic", "v2like"],
        help="Graph merge engine: classic or v2like (async candidate verify + repair).",
    )
    parser.add_argument(
        "--merge-decisions-log-path",
        type=str,
        default=None,
        help="JSONL path for merge decision traces (v2like).",
    )
    parser.add_argument(
        "--v2-top-k",
        type=int,
        default=10,
        help="Top-k candidates for v2like merge.",
    )
    parser.add_argument(
        "--v2-judge-concurrency",
        type=int,
        default=6,
        help="Async LLM judge concurrency for v2like merge.",
    )
    parser.add_argument(
        "--v2-disable-llm-judge",
        action="store_true",
        help="Disable LLM judge in v2like merge (debug only).",
    )
    parser.add_argument(
        "--v2-disable-bridge-merge",
        action="store_true",
        help="Disable repair/bridge merge pass in v2like merge.",
    )
    parser.add_argument(
        "--baseline-graph-path",
        type=str,
        default=None,
        help="Optional baseline graph pickle for merge quality comparison.",
    )
    parser.add_argument(
        "--enforce-merge-gates",
        action="store_true",
        help="Fail run when merge quality gates detect degradation.",
    )
    parser.add_argument(
        "--merge-quality-report-path",
        type=str,
        default=None,
        help="Path to save merge quality report JSON.",
    )
    parser.add_argument(
        "--max-relation-drop-ratio",
        type=float,
        default=0.45,
        help="Maximum allowed relation evidence drop ratio vs baseline.",
    )

    args = parser.parse_args()

    book_path = Path(args.book_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = OUTPUT_DIR / "bookgraph.pkl"

    chapter_limit = None
    if args.debug:
        requested_limit = args.max_chapters if args.max_chapters > 0 else 5
        chapter_limit = min(requested_limit, 5)
        logger.info(
            "Debug mode: processing only %s chapters (hard cap 5)", chapter_limit
        )
    elif args.max_chapters > 0:
        chapter_limit = args.max_chapters
        logger.info("Chapter limit enabled: %s", chapter_limit)

    try:
        # Инициализация LLM
        logger.info("Инициализация LLM...")
        llm = create_llm(llm_type=args.llm_type)

        # Создание промптов
        logger.info("Создание промптов...")
        json_parser = JsonOutputParser()
        extractor = ExtractNames(llm=llm)
        extractor_relations = ExtractRelations(llm=llm)
        verificator = Verification(llm=llm, parser=json_parser)

        # Инициализация сервисов
        logger.info("Инициализация сервисов...")
        verification_service = VerificationService(verificator)
        node_processor = NodeProcessor(verification_service)
        relation_processor = RelationProcessor()

        extraction_service = ExtractionService(
            extractor=extractor,
            extractor_relations=extractor_relations,
        )

        # Этап 1: Загрузка и разбиение книги на ГЛАВЫ
        # Важно: разбиение происходит на главы (большие части по разделителям),
        # а не на маленькие чанки. Маленькие чанки создаются только для vectorstore.
        texts = None
        if not args.skip_extraction:
            logger.info("Этап 1: Загрузка и разбиение книги на главы...")
            book_text = load_book(book_path)
            # split_book по умолчанию разбивает на главы (split_into_chunks=False)
            texts = split_book(book_text)
            if chapter_limit:
                texts = texts[:chapter_limit]
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
            extraction_outputs = extraction_service.extract_from_texts(
                texts, concurrency=PARALLEL_CONCURRENCY
            )
            logger.info(f"Обработано {len(extraction_outputs)} глав")
        else:
            logger.info(
                "Пропуск этапа экстракции (используются существующие результаты)"
            )

        # Этап 3: Построение графа из результатов экстракции
        logger.info("Stage 3: build graph (engine=%s)...", args.merge_engine)
        if args.merge_engine == "v2like":
            decisions_log_path = (
                Path(args.merge_decisions_log_path)
                if args.merge_decisions_log_path
                else output_path.with_suffix(".merge_decisions.jsonl")
            )
            decisions_log_path.parent.mkdir(parents=True, exist_ok=True)

            pipeline = VerificationPipelineV2(
                results_dir=RESULTS_DIR,
                output_path=output_path,
                logs_path=decisions_log_path,
                top_k=max(1, int(args.v2_top_k)),
                llm_type=args.llm_type,
                llm_enabled=not args.v2_disable_llm_judge,
                max_files=chapter_limit if chapter_limit and chapter_limit > 0 else None,
                enable_bridge_merge=not args.v2_disable_bridge_merge,
                judge_concurrency=max(1, int(args.v2_judge_concurrency)),
            )
            book_graph = pipeline.run()
            logger.info(
                "v2like merge completed: nodes=%s relations=%s logs=%s",
                len(book_graph.nodes.nodes),
                len(book_graph.relationships.relationships),
                decisions_log_path,
            )
        else:
            batch_payload_dir = RESULTS_DIR / "_batch_payloads"
            batch_files = sorted(batch_payload_dir.glob("batch_*.json"))
            if not batch_files:
                raise FileNotFoundError(
                    "Batch payloads are required for merge but were not found in "
                    f"{batch_payload_dir}"
                )

            max_batches = None
            if chapter_limit and chapter_limit > 0:
                batch_size = max(1, int(EXTRACTION_CHAPTER_BATCH_SIZE))
                max_batches = max(1, math.ceil(chapter_limit / batch_size))

            batch_builder = BatchGraphBuilder(
                node_processor=node_processor,
                relation_processor=relation_processor,
            )
            all_book_nodes, relation_graphs, batch_reports = (
                batch_builder.build_graph_from_batch_payloads(
                    batch_payload_dir=batch_payload_dir,
                    max_batches=max_batches,
                )
            )
            logger.info(
                "Classic merge via batch payloads completed: batches=%s nodes=%s relations=%s",
                len(batch_reports),
                len(all_book_nodes.nodes),
                len(relation_graphs.relationships),
            )
            logger.info(
                f"Graph built: nodes={len(all_book_nodes.nodes)} "
                f"relations={len(relation_graphs.relationships)}"
            )
            book_graph = BookGraph(nodes=all_book_nodes, relationships=relation_graphs)

        baseline_graph = None
        if args.baseline_graph_path:
            baseline_path = Path(args.baseline_graph_path)
            if not baseline_path.exists():
                raise FileNotFoundError(f"Baseline graph not found: {baseline_path}")
            baseline_graph = FileManager.load_pickle(baseline_path)

        if baseline_graph is not None or args.enforce_merge_gates:
            quality_report = evaluate_merge_quality(
                graph=book_graph,
                baseline_graph=baseline_graph,
                max_relation_drop_ratio=float(args.max_relation_drop_ratio),
            )
            quality_report_path = (
                Path(args.merge_quality_report_path)
                if args.merge_quality_report_path
                else output_path.with_suffix(".merge_quality.json")
            )
            quality_report_path.parent.mkdir(parents=True, exist_ok=True)
            quality_report_path.write_text(
                json.dumps(quality_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Merge quality report saved to %s", quality_report_path)
            if quality_report.get("warnings"):
                logger.warning("Merge quality warnings: %s", quality_report["warnings"])
            if args.enforce_merge_gates and not quality_report.get("passed", False):
                raise RuntimeError(
                    "Merge quality gates failed: "
                    + "; ".join(quality_report.get("violations", []))
                )

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
                expected_node_facts, expected_relation_facts = expected_graph_fact_counts(
                    book_graph
                )
                graph_docs = [
                    doc
                    for doc in all_documents
                    if doc.metadata.get("source") in {"nodes", "relations"}
                ]
                graph_gate_stats = validate_graph_documents(
                    graph_docs,
                    expected_node_facts=expected_node_facts,
                    expected_relation_facts=expected_relation_facts,
                )
                logger.info(
                    "Graph quality gate passed: total=%s, nodes=%s, relations=%s",
                    graph_gate_stats["total_docs"],
                    graph_gate_stats["nodes_docs"],
                    graph_gate_stats["relations_docs"],
                )

                # Отделяем главы от остальных документов
                logger.info("Разделение глав и остальных документов...")
                chapters = preparer.get_chapters(all_documents)
                other_documents = [doc for doc in all_documents if doc not in chapters]
                chapter_gate_stats = validate_chapters(chapters)
                logger.info(
                    "Chapters quality gate passed: chapters_count=%s",
                    chapter_gate_stats["chapters_count"],
                )
                other_gate_stats = validate_non_chapter_documents(other_documents)
                logger.info(
                    "Non-chapter quality gate passed: %s",
                    other_gate_stats,
                )
                logger.info(
                    f"Найдено {len(chapters)} глав и {len(other_documents)} других документов"
                )

                # Разбиваем на мелкие чанки только остальные документы (не главы)
                logger.info("Разбиение документов (кроме глав) на мелкие чанки...")
                split_docs = preparer.split_to_small_chunks(other_documents)
                split_gate_stats = validate_non_chapter_documents(split_docs)
                logger.info("Split documents quality gate passed: %s", split_gate_stats)

                # Incremental rebuild by changed source_id hashes.
                manifest_path = OUTPUT_DIR / "preprocess_manifest.json"
                current_hashes = _source_hash_map(split_docs)
                previous_hashes = _load_manifest(manifest_path).get("source_hashes", {})
                changed_source_ids = _detect_changed_source_ids(
                    current_hashes=current_hashes,
                    previous_hashes=previous_hashes,
                )
                if args.vectorstore_force_reload:
                    changed_source_ids = sorted(current_hashes.keys())

                if changed_source_ids:
                    split_docs = [
                        d for d in split_docs if d.metadata.get("source_id") in changed_source_ids
                    ]
                    chapters = [
                        c for c in chapters if c.metadata.get("source_id") in changed_source_ids
                    ]
                    logger.info(
                        "Incremental mode: changed source_ids=%s, docs=%s, chapters=%s",
                        len(changed_source_ids),
                        len(split_docs),
                        len(chapters),
                    )
                else:
                    logger.info("Incremental mode: no changes detected, skip indexing")
                    split_docs = []
                    chapters = []

                # Загружаем чанки в векторное хранилище
                if split_docs:
                    logger.info("Загрузка чанков в векторное хранилище...")
                    loader = VectorStoreLoader()
                    loader.load_documents(
                        documents=split_docs,
                        output_pickle_path=vectorstore_pickle_path,
                        force_reload=args.vectorstore_force_reload,
                        changed_source_ids=changed_source_ids,
                    )
                    logger.info("Чанки успешно загружены в векторное хранилище!")
                    logger.info(f"Pickle файл сохранен в: {vectorstore_pickle_path}")
                else:
                    logger.info("Чанки не изменились, загрузка в vectorstore пропущена")

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

                _save_manifest(manifest_path, current_hashes)
                logger.info("Incremental manifest saved: %s", manifest_path)

            except Exception as e:
                logger.error(
                    f"Ошибка при загрузке в векторное хранилище: {e}", exc_info=True
                )
                # Не прерываем выполнение, только логируем ошибку

        # Этап 5: Snapshot export (опционально)
        if args.export_snapshot:
            try:
                from tools.preprocess_book.snapshot import export_snapshot, validate_snapshot

                snapshot_output_dir = (
                    Path(args.snapshot_output_dir)
                    if args.snapshot_output_dir
                    else Path("tools/preprocess_book/_snapshot")
                )
                snapshot_max_files = args.snapshot_max_files
                if snapshot_max_files <= 0 and chapter_limit:
                    snapshot_max_files = chapter_limit

                logger.info("Этап 5: Экспорт snapshot в %s...", snapshot_output_dir)
                export_stats = export_snapshot(
                    output_dir=snapshot_output_dir,
                    book_path=book_path,
                    max_files=snapshot_max_files,
                    results_dir=RESULTS_DIR,
                    graph_nodes_dir=GRAPH_NODES_DIR,
                    graph_relations_dir=GRAPH_RELATIONS_DIR,
                    merged_graph_path=output_path,
                )
                logger.info("Snapshot export завершен: %s", export_stats)

                if args.snapshot_validate:
                    validation = validate_snapshot(snapshot_output_dir)
                    validation_path = snapshot_output_dir / "validation_report.json"
                    validation_path.write_text(
                        json.dumps(validation, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(
                        "Snapshot validation status=%s report=%s",
                        validation.get("status"),
                        validation_path,
                    )
            except Exception as e:
                logger.error("Ошибка при snapshot export: %s", e, exc_info=True)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
