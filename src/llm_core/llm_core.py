import os

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from src.config import settings as settings

if settings.LLM_TYPE == "chatgpt":
    token = os.getenv("OPENAI_API_KEY")
    if not token:
        raise RuntimeError(
            "OPENAI_API_KEY is required for LLM_TYPE=chatgpt. "
            "Set it in environment variables."
        )
    llm = ChatOpenAI(
        model=settings.OPENAI_API_MODEL,
        temperature=0,
        max_retries=2,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_key=token,
    )
elif settings.LLM_TYPE == "deepseek":
    token = os.getenv("DEEPSEEK_API_KEY")
    if not token:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for LLM_TYPE=deepseek. "
            "Set it in environment variables."
        )
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=token,
    )
