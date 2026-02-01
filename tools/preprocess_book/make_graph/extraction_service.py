import logging
from typing import List

from langchain_core.documents import Document
from tools.preprocess_book.config.preprocess_settings import RESULTS_DIR
from tools.preprocess_book.storage.storage import FileManager
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ExtractionService:
    """Сервис для извлечения сущностей и отношений из текстов."""

    def __init__(self, extractor, extractor_relations=None):
        """
        Args:
            extractor: Экстрактор для извлечения сущностей и отношений
            extractor_relations: Экстрактор для отношений (опционально, если отличается от extractor)
        """
        self.extractor = extractor
        self.extractor_relations = extractor_relations or extractor

    def extract_from_texts(self, texts: List[Document]) -> List[str]:
        """
        Извлекает сущности и отношения из текстов и сохраняет результаты.

        Args:
            texts: Список Document объектов с текстами для обработки

        Returns:
            Список суммаризаций текстов
        """
        all_summarizations = []

        for text in tqdm(texts, desc="Извлечение сущностей"):
            try:
                source_id = text.metadata.get("source_id")
                if source_id is None:
                    logger.warning("Текст без source_id пропущен")
                    continue

                source_filename = FileManager.get_source_filename(source_id)

                # Извлечение нод и отношений
                res = self.extractor.invoke(text=text.page_content)

                # Сохранение результатов экстракции
                FileManager.save_pickle(res, RESULTS_DIR / f"{source_filename}.pkl")

                # Сохранение суммаризации если есть
                if "summarization" in res:
                    all_summarizations.append(res["summarization"])
                else:
                    logger.warning(
                        f"Результат для {source_id} не содержит суммаризации"
                    )
                    all_summarizations.append("")

            except Exception as e:
                logger.error(
                    f"Ошибка при обработке source_id {source_id}: {e}", exc_info=True
                )
                continue

        return all_summarizations
