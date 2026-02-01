"""
Модуль с промптами для экстракции и верификации.
"""

from tools.preprocess_book.prompts.extract_names import ExtractNames, system_prompt as extract_system_prompt
from tools.preprocess_book.prompts.verificator import Verification, system_prompt as verificator_system_prompt

__all__ = [
    "ExtractNames",
    "extract_system_prompt",
    "Verification",
    "verificator_system_prompt",
]
