"""
Сервис для создания суммаризаций глав книги.

Модуль содержит класс SummarizationService для обработки
списка глав и создания текстовых файлов с суммаризациями.
"""

import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document

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
    
    def create_summaries(
        self, 
        texts: List[Document], 
        output_dir: Path
    ) -> List[str]:
        """
        Создает суммаризации для списка текстов и сохраняет их в файлы.
        
        Args:
            texts: Список Document объектов с текстами глав для обработки
            output_dir: Директория для сохранения файлов суммаризаций
            
        Returns:
            Список созданных суммаризаций (в порядке обработки текстов)
            
        Raises:
            ValueError: Если output_dir не является директорией или не может быть создан
        """
        # Создаем директорию если она не существует
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not output_dir.is_dir():
            raise ValueError(f"output_dir должен быть директорией: {output_dir}")
        
        all_summarizations = []
        
        for text in tqdm(texts, desc="Создание суммаризаций"):
            try:
                source_id = text.metadata.get("source_id")
                if source_id is None:
                    logger.warning("Текст без source_id пропущен")
                    all_summarizations.append("")
                    continue
                
                # Создаем суммаризацию
                messages = self.summarizer.make_user_prompt(text.page_content)
                result = self.summarizer.invoke(**messages)
                
                # Извлекаем текст суммаризации
                if isinstance(result, str):
                    summary_text = result
                elif hasattr(result, 'content'):
                    summary_text = result.content
                else:
                    logger.warning(f"Неожиданный тип результата для source_id {source_id}: {type(result)}")
                    summary_text = str(result)
                
                # Сохраняем в файл
                summary_file = output_dir / f"{source_id}.txt"
                with open(summary_file, "w", encoding="utf-8") as file:
                    file.write(summary_text)
                
                all_summarizations.append(summary_text)
                logger.debug(f"Создана суммаризация для source_id {source_id}")
                
            except Exception as e:
                logger.error(
                    f"Ошибка при создании суммаризации для source_id {source_id}: {e}",
                    exc_info=True
                )
                all_summarizations.append("")
                continue
        
        logger.info(f"Создано {len([s for s in all_summarizations if s])} суммаризаций из {len(texts)} глав")
        return all_summarizations
