import os
import logging
from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from tools.preprocess_book.config.preprocess_settings import (
    LLM_TYPE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_API_MODEL,
)


def create_llm(
    llm_type: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    max_retries: int = 2,
    **kwargs
) -> BaseLanguageModel:
    """
    Создает экземпляр LLM в зависимости от типа.
    
    Args:
        llm_type: Тип LLM ("deepseek" или "chatgpt"). Если None, берется из настроек
        model: Название модели (опционально)
        temperature: Температура модели
        max_retries: Количество повторных попыток при ошибке
        **kwargs: Дополнительные параметры для LLM
        
    Returns:
        Экземпляр BaseLanguageModel
        
    Raises:
        ValueError: Если указан неподдерживаемый тип LLM
        ValueError: Если отсутствует API ключ
    """
    if llm_type is None:
        llm_type = LLM_TYPE
    
    if llm_type == "deepseek":
        # DeepSeek client in langchain uses OpenAI-compatible transport under the hood.
        # Suppress internal openai logger noise so deepseek runs stay provider-clean in logs.
        logging.getLogger("openai").setLevel(logging.ERROR)
        api_key = kwargs.get("api_key") or DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY не установлен в переменных окружения")
        
        return ChatDeepSeek(
            model=model or "deepseek-chat",
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=max_retries,
            api_key=api_key,
            api_base=kwargs.get("api_base") or DEEPSEEK_API_BASE,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in {"api_key", "api_base", "openai_api_base"}
            }
        )
    
    elif llm_type == "chatgpt" or llm_type == "openai":
        api_key = kwargs.get("api_key") or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY не установлен в переменных окружения")
        
        return ChatOpenAI(
            model=model or OPENAI_API_MODEL,
            temperature=temperature,
            max_retries=max_retries,
            openai_api_base=kwargs.get("openai_api_base") or OPENAI_API_BASE,
            openai_api_key=api_key,
            **{k: v for k, v in kwargs.items() if k not in ["api_key", "openai_api_base"]}
        )
    
    else:
        raise ValueError(f"Неподдерживаемый тип LLM: {llm_type}. Используйте 'deepseek' или 'chatgpt'")
