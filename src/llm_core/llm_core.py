import os

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from src.config import settings as settings

if settings.LLM_TYPE == "chatgpt":
    token = os.getenv("OPENAI_API_KEY")
    if not token:
        token = input("Введите токен OpenAI:")
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_retries=2,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_key=token,
    )
elif settings.LLM_TYPE == "deepseek":
    token = os.getenv("DEEPSEEK_API_KEY")
    if not token:
        token = input("Введите токен DeepSeek:")
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=token,
        # other params...
    )
