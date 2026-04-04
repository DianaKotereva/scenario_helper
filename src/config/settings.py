import os

import yaml
from dotenv import load_dotenv

load_dotenv()

# OUTPUT_FILE_SAVE_PATH: Path = Path(os.getenv("OUTPUT_FILE_SAVE_PATH"))

BM25_K: int = int(os.getenv("BM25_K", 10))
ES_RETRIEVER_K: int = int(os.getenv("ES_RETRIEVER_K", 10))
ES_SIMILARITY_THRESHOLD: float = float(os.getenv("ES_SIMILARITY_THRESHOLD", 0.1))

ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "shadow_and_flame_")
ES_BATCH_SIZE: int = int(os.getenv("ES_BATCH_SIZE", 100))
ES_PICKLE_DOCUMENTS_PATH: str = os.getenv(
    "ES_PICKLE_DOCUMENTS_PATH", "data/all_langchain_chunks.pkl"
)
ES_TIMEOUT_RAW: str = os.getenv("ES_TIMEOUT", "").strip()
ES_TIMEOUT: int | None = (
    None
    if ES_TIMEOUT_RAW.lower() in {"", "none", "null", "off", "false", "0"}
    else int(ES_TIMEOUT_RAW)
)
ES_MAX_RETRIES: int = int(os.getenv("ES_MAX_RETRIES", 5))
ES_RETRY_ON_TIMEOUT: bool = os.getenv("ES_RETRY_ON_TIMEOUT", "True") == "True"

GRAPH_PICKLE_PATH: str = os.getenv("GRAPH_PICKLE_PATH", "data/bookgraph.pkl")

# Настройки для получения глав
CHAPTERS_INDEX_NAME: str = os.getenv("CHAPTERS_INDEX_NAME", f"{ES_INDEX_NAME}chapters")
INCLUDE_FULL_CHAPTERS: bool = os.getenv("INCLUDE_FULL_CHAPTERS", "True") == "True"
MAX_CHAPTERS_IN_CONTEXT: int = int(os.getenv("MAX_CHAPTERS_IN_CONTEXT", "3"))
MAX_CHAPTER_TOKENS: int = int(os.getenv("MAX_CHAPTER_TOKENS", "15000"))
SNAPSHOT_DIR: str = os.getenv("SNAPSHOT_DIR", "tools/preprocess_book/_snapshot25_p0")

# Гибридный retrieval (P1)
PHASE_A_RECALL_K: int = int(os.getenv("PHASE_A_RECALL_K", "20"))
PHASE_C_FALLBACK_K: int = int(os.getenv("PHASE_C_FALLBACK_K", "30"))
PHASE_B_MAX_ENTITIES: int = int(os.getenv("PHASE_B_MAX_ENTITIES", "10"))
PHASE_B_MAX_RELATIONS: int = int(os.getenv("PHASE_B_MAX_RELATIONS", "20"))
PHASE_B_MAX_CHAPTERS: int = int(os.getenv("PHASE_B_MAX_CHAPTERS", "5"))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_MODEL: str = os.getenv("OPENAI_API_MODEL", "gpt-4.1-nano")
OPENAI_EMB_MODEL: str = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
LLM_TYPE: str = os.getenv("LLM_TYPE", "deepseek")


LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECt: str = os.getenv("LANGSMITH_PROJECt")

MAX_N_ITERATIONS: int = int(os.getenv("MAX_N_ITERATIONS", 2))

FORCE_RELOAD: bool = os.getenv("FORCE_RELOAD", "False") == "True"
USE_FAISS: bool = os.getenv("USE_FAISS", "True") == "True"
FAISS_PATH: str = os.getenv("FAISS_PATH", "data/vector_store.faiss")

with open("src/config/open_search_settings.yaml", "r", encoding="utf-8") as f:
    open_search_settings = yaml.safe_load(f)
