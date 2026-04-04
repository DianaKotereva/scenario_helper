"""
Сервис для создания суммаризаций глав книги.

Модуль содержит класс SummarizationService для обработки
списка глав и создания текстовых файлов с суммаризациями.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from tools.preprocess_book.config.preprocess_settings import PARALLEL_CONCURRENCY

logger = logging.getLogger(__name__)


class SummarizationService:
    """
    Сервис для создания суммаризаций глав книги.

    Обрабатывает список Document объектов (глав книги) и создает
    текстовые файлы с суммаризациями для каждой главы.
    """

    def __init__(self, summarizer):
        """
        Инициализирует SummarizationService.

        Args:
            summarizer: Экземпляр SummarizationPrompt для создания суммаризаций
        """
        self.summarizer = summarizer

    async def create_summaries_async(
        self, texts: List[Document], output_dir: Path, concurrency: int = 5
    ) -> List[str]:
        """
        Создает суммаризации параллельно используя batch() метод.

        Args:
            texts: Список Document объектов с текстами глав для обработки
            output_dir: Директория для сохранения файлов суммаризаций
            concurrency: Количество одновременных запросов к LLM

        Returns:
            Список созданных суммаризаций (в порядке обработки текстов)

        Raises:
            ValueError: Если output_dir не является директорией или не может быть создан
        """
        # Создаем директорию если она не существует
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_dir.is_dir():
            raise ValueError(f"output_dir должен быть директорией: {output_dir}")

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

            inputs.append(self.summarizer.make_user_prompt(text.page_content))
            source_ids.append(source_id)
            text_indices.append(len(inputs) - 1)  # Индекс в списке inputs

        if not inputs:
            logger.warning("Нет валидных текстов для создания суммаризаций")
            return [""] * len(texts)

        # Параллельная обработка через batch()
        logger.info(
            f"Начинаем параллельную обработку {len(inputs)} суммаризаций с concurrency={concurrency}"
        )
        results = await self.summarizer.batch(inputs, concurrency=concurrency)

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

                # Извлекаем текст суммаризации
                if isinstance(result, str):
                    summary_text = result
                elif hasattr(result, "content"):
                    summary_text = result.content
                else:
                    logger.warning(
                        f"Неожиданный тип результата для source_id {source_ids[text_indices[idx]]}: {type(result)}"
                    )
                    summary_text = str(result)

                # Сохраняем в файл
                source_id = source_ids[text_indices[idx]]
                summary_file = output_dir / f"{source_id}.txt"
                with open(summary_file, "w", encoding="utf-8") as file:
                    file.write(summary_text)

                all_summarizations.append(summary_text)
                logger.debug(f"Создана суммаризация для source_id {source_id}")
                if isinstance(source_id, int) and source_id % 5 == 0:
                    logger.info(
                        "Summaries progress marker: chapter source_id=%s",
                        source_id,
                    )

            except Exception as e:
                source_id = (
                    source_ids[text_indices[idx]]
                    if text_indices[idx] < len(source_ids)
                    else "unknown"
                )
                logger.error(
                    f"Ошибка при создании суммаризации для source_id {source_id}: {e}",
                    exc_info=True,
                )
                all_summarizations.append("")
                continue

        logger.info(
            f"Создано {len([s for s in all_summarizations if s])} суммаризаций из {len(texts)} глав"
        )
        return all_summarizations

    def create_summaries(
        self, texts: List[Document], output_dir: Path, concurrency: Optional[int] = None
    ) -> List[str]:
        """
        Создает суммаризации для списка текстов и сохраняет их в файлы.
        Синхронная обертка для create_summaries_async.

        Args:
            texts: Список Document объектов с текстами глав для обработки
            output_dir: Директория для сохранения файлов суммаризаций
            concurrency: Количество одновременных запросов к LLM (по умолчанию из настроек)

        Returns:
            Список созданных суммаризаций (в порядке обработки текстов)

        Raises:
            ValueError: Если output_dir не является директорией или не может быть создан
        """
        concurrency = concurrency or PARALLEL_CONCURRENCY
        return asyncio.run(self.create_summaries_async(texts, output_dir, concurrency))
