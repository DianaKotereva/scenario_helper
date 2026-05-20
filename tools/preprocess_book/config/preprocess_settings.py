import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Базовые пути
BASE_DIR = Path(__file__).parent.parent.parent.parent
TOOLS_DIR = BASE_DIR / "tools"
PREPROCESS_DIR = TOOLS_DIR / "preprocess_book"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
PREPROCESS_WORK_DIR = PROCESSED_DATA_DIR / "preprocess_work"

# Директории для обработки книги
BOOK_INPUT_DIR = Path(os.getenv("BOOK_INPUT_DIR", str(DATA_DIR)))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PREPROCESS_WORK_DIR / "results")))
GRAPH_NODES_DIR = Path(
    os.getenv("GRAPH_NODES_DIR", str(PREPROCESS_WORK_DIR / "graph_nodes"))
)
GRAPH_RELATIONS_DIR = Path(
    os.getenv("GRAPH_RELATIONS_DIR", str(PREPROCESS_WORK_DIR / "graph_relations"))
)
SUMMARIES_DIR = Path(os.getenv("SUMMARIES_DIR", str(PREPROCESS_WORK_DIR / "summaries")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(DATA_DIR)))

# Настройки LLM
LLM_TYPE = os.getenv("LLM_TYPE", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4.1-nano")

# Настройки сплиттера
TEXT_SPLITTER_SEPARATORS = ["========== ", "***"]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Настройки обработки
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
VERIFICATION_LAST_N = int(os.getenv("VERIFICATION_LAST_N", -10))
EXTRACTION_CHAPTER_BATCH_SIZE = int(os.getenv("EXTRACTION_CHAPTER_BATCH_SIZE", "5"))
EXTRACTION_VALIDATION_RETRY_COUNT = int(
    os.getenv("EXTRACTION_VALIDATION_RETRY_COUNT", "3")
)
EXTRACTION_USE_SMALL_CHUNKS = _env_bool("EXTRACTION_USE_SMALL_CHUNKS", True)
EXTRACTION_SMALL_CHUNK_SIZE = int(os.getenv("EXTRACTION_SMALL_CHUNK_SIZE", "6000"))
EXTRACTION_SMALL_CHUNK_OVERLAP = int(
    os.getenv("EXTRACTION_SMALL_CHUNK_OVERLAP", "250")
)
V2_VERIFICATION_LAST_N = int(os.getenv("V2_VERIFICATION_LAST_N", "-50"))
BATCH_STRICT_MERGE = _env_bool("BATCH_STRICT_MERGE", True)

# Настройки для vectorstore
VECTORSTORE_CHUNK_SIZE = int(os.getenv("VECTORSTORE_CHUNK_SIZE", 512))
VECTORSTORE_PICKLE_PATH = Path(
    os.getenv("VECTORSTORE_PICKLE_PATH", str(DATA_DIR / "all_langchain_chunks.pkl"))
)

# Настройки для индекса глав
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "shadow_and_flame_")
CHAPTERS_INDEX_NAME = os.getenv("CHAPTERS_INDEX_NAME", f"{ES_INDEX_NAME}chapters")
CHAPTERS_PICKLE_PATH = Path(
    os.getenv("CHAPTERS_PICKLE_PATH", str(DATA_DIR / "chapters.pkl"))
)

# Настройки параллельности
PARALLEL_CONCURRENCY = int(
    os.getenv("PARALLEL_CONCURRENCY", "200")
)  # Количество одновременных запросов к LLM

# Создание директорий если они не существуют
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_NODES_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_RELATIONS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Pre-NER helper settings (hard-disabled; cannot be overridden via env)
ENABLE_PRE_NER_HELPER = False
# Master switch: disables any pre-NER stage (Natasha + regex helper + coverage retry inputs).
PRE_NER_ENABLED = False
PRE_NER_MAX_ENTITIES_PER_BATCH = 120
PRE_NER_COVERAGE_RETRY = False
PRE_NER_USE_NATASHA = False

