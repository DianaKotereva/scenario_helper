"""
Модуль для создания суммаризаций глав книги.

Содержит классы для генерации повествовательных суммаризаций
на основе текста глав книги с использованием LLM.
"""

from .summarization_service import SummarizationService
from .prompts import SummarizationPrompt

__all__ = [
    "SummarizationService",
    "SummarizationPrompt",
]
