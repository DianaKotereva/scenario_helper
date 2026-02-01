import ast
import logging
import os
from typing import Tuple

from src.utils.graph_search import AllBookNodes, AllBooksEdges
from tools.preprocess_book.config.preprocess_settings import (
    GRAPH_NODES_DIR,
    GRAPH_RELATIONS_DIR,
    RESULTS_DIR,
)
from tools.preprocess_book.make_graph.node_processor import NodeProcessor
from tools.preprocess_book.make_graph.relation_processor import RelationProcessor
from tools.preprocess_book.storage.storage import FileManager
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(
        self, node_processor: NodeProcessor, relation_processor: RelationProcessor
    ):
        self.node_processor = node_processor
        self.relation_processor = relation_processor

    def build_graph_from_results(self) -> Tuple[AllBookNodes, AllBooksEdges]:
        """
        Строит граф из сохраненных результатов экстракции.

        Returns:
            Кортеж (AllBookNodes, AllBooksEdges) с построенным графом
        """
        all_book_nodes = AllBookNodes()
        relation_graphs = AllBooksEdges()

        # Получаем и сортируем файлы результатов
        if not RESULTS_DIR.exists():
            logger.error(f"Директория результатов не найдена: {RESULTS_DIR}")
            return all_book_nodes, relation_graphs

        all_results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl")]

        if not all_results_files:
            logger.warning(f"Не найдено файлов результатов в {RESULTS_DIR}")
            return all_book_nodes, relation_graphs

        # Сортируем файлы по номеру source_id
        def extract_source_id(filename: str) -> int:
            """Извлекает source_id из имени файла для сортировки."""
            try:
                # Пытаемся извлечь число из имени файла
                name_without_ext = filename.replace(".pkl", "")
                # Если это формат типа "1_2_3", берем первое число
                parts = name_without_ext.split("_")
                return int(parts[0]) if parts else 0
            except (ValueError, IndexError):
                return 0

        all_results_files = sorted(all_results_files, key=extract_source_id)

        for source_txt in tqdm(all_results_files, desc="Построение графа"):
            try:
                # Безопасное извлечение source_id из имени файла
                source_id_str = source_txt.replace(".pkl", "")
                try:
                    # Пытаемся распарсить как кортеж
                    source_id = ast.literal_eval(source_id_str)
                    if not isinstance(source_id, tuple):
                        source_id = (source_id,)
                except (ValueError, SyntaxError):
                    # Если не получается, пытаемся извлечь числа
                    parts = source_id_str.split("_")
                    source_id = tuple(int(p) for p in parts if p.isdigit())
                    if not source_id:
                        logger.warning(f"Не удалось извлечь source_id из {source_txt}")
                        continue

                # Загружаем результаты экстракции
                result_path = RESULTS_DIR / source_txt
                if not result_path.exists():
                    logger.warning(f"Файл результатов не найден: {result_path}")
                    continue

                res = FileManager.load_pickle(result_path)

                # Валидация структуры результата
                if not isinstance(res, dict):
                    logger.warning(
                        f"Неверный формат результата для {source_txt}: ожидается dict"
                    )
                    continue

                nodes = res.get("nodes", [])
                relations = res.get("relations", [])

                # Обрабатываем ноды и отношения
                all_book_nodes = self.node_processor.process_input(
                    input_data=nodes,
                    rel_inputs=relations,
                    source_id=source_id,
                    all_book_nodes=all_book_nodes,
                )

                relation_graphs = self.relation_processor.process_relations(
                    all_book_nodes=all_book_nodes,
                    rel_inputs=relations,
                    rel_graphs=relation_graphs,
                    source_id=source_id,
                )

                # Сохраняем промежуточные результаты
                FileManager.save_pickle(all_book_nodes, GRAPH_NODES_DIR / source_txt)
                FileManager.save_pickle(
                    relation_graphs, GRAPH_RELATIONS_DIR / source_txt
                )

            except Exception as e:
                logger.error(f"Ошибка при обработке {source_txt}: {e}", exc_info=True)
                continue

        return all_book_nodes, relation_graphs
