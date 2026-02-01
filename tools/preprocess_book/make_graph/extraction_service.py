import asyncio
import logging
from typing import List, Optional

from langchain_core.documents import Document
from tools.preprocess_book.config.preprocess_settings import (
    PARALLEL_CONCURRENCY,
    RESULTS_DIR,
)
from tools.preprocess_book.storage.storage import FileManager

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

    async def extract_from_texts_async(
        self, texts: List[Document], concurrency: int = 5
    ) -> List[str]:
        """
        Извлекает сущности и отношения из текстов параллельно используя batch() метод.

        Args:
            texts: Список Document объектов с текстами для обработки
            concurrency: Количество одновременных запросов к LLM

        Returns:
            Список суммаризаций текстов
        """
        # Подготавливаем входные данные для batch
        inputs = []
        source_ids = []
        text_indices = []  # Для сохранения порядка и обработки пропущенных текстов

        for idx, text in enumerate(texts):
            source_id = text.metadata.get("source_id")
            if source_id is None:
                logger.warning(f"Текст на позиции {idx} без source_id пропущен")
                text_indices.append(None)
                continue

            inputs.append(self.extractor.make_user_prompt(text=text.page_content))
            source_ids.append(source_id)
            text_indices.append(len(inputs) - 1)  # Индекс в списке inputs

        if not inputs:
            logger.warning("Нет валидных текстов для экстракции")
            return []

        # Параллельная обработка через batch()
        logger.info(
            f"Начинаем параллельную экстракцию {len(inputs)} текстов с concurrency={concurrency}"
        )
        results = await self.extractor.batch(inputs, concurrency=concurrency)

        # Сохранение результатов
        all_summarizations = []
        result_idx = 0

        for idx, text in enumerate(texts):
            if text_indices[idx] is None:
                all_summarizations.append("")
                continue

            try:
                result = results[result_idx]
                result_idx += 1
                source_id = source_ids[text_indices[idx]]
                source_filename = FileManager.get_source_filename(source_id)

                # Сохранение результатов экстракции
                FileManager.save_pickle(result, RESULTS_DIR / f"{source_filename}.pkl")

                # Сохранение суммаризации если есть
                if isinstance(result, dict) and "summarization" in result:
                    all_summarizations.append(result["summarization"])
                else:
                    logger.warning(
                        f"Результат для {source_id} не содержит суммаризации"
                    )
                    all_summarizations.append("")

            except Exception as e:
                source_id = (
                    source_ids[text_indices[idx]]
                    if text_indices[idx] < len(source_ids)
                    else "unknown"
                )
                logger.error(
                    f"Ошибка при обработке source_id {source_id}: {e}", exc_info=True
                )
                all_summarizations.append("")
                continue

        logger.info(
            f"Обработано {len([s for s in all_summarizations if s])} текстов из {len(texts)} глав"
        )
        return all_summarizations

    def extract_from_texts(
        self, texts: List[Document], concurrency: Optional[int] = None
    ) -> List[str]:
        """
        Извлекает сущности и отношения из текстов и сохраняет результаты.
        Синхронная обертка для extract_from_texts_async.

        Args:
            texts: Список Document объектов с текстами для обработки
            concurrency: Количество одновременных запросов к LLM (по умолчанию из настроек)

        Returns:
            Список суммаризаций текстов
        """
        concurrency = concurrency or PARALLEL_CONCURRENCY
        return asyncio.run(self.extract_from_texts_async(texts, concurrency))
