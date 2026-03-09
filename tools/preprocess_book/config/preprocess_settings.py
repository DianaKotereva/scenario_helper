import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).parent.parent.parent.parent
TOOLS_DIR = BASE_DIR / "tools"
PREPROCESS_DIR = TOOLS_DIR / "preprocess_book"
DATA_DIR = BASE_DIR / "data"

# Директории для обработки книги
BOOK_INPUT_DIR = Path(os.getenv("BOOK_INPUT_DIR", str(DATA_DIR)))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PREPROCESS_DIR / "results")))
GRAPH_NODES_DIR = Path(
    os.getenv("GRAPH_NODES_DIR", str(PREPROCESS_DIR / "graph_nodes"))
)
GRAPH_RELATIONS_DIR = Path(
    os.getenv("GRAPH_RELATIONS_DIR", str(PREPROCESS_DIR / "graph_relations"))
)
SUMMARIES_DIR = Path(os.getenv("SUMMARIES_DIR", str(PREPROCESS_DIR / "summaries")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(DATA_DIR)))

# Настройки LLM
LLM_TYPE = os.getenv("LLM_TYPE", "deepseek")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
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
GRAPH_BUILD_CONCURRENCY = int(
    os.getenv("GRAPH_BUILD_CONCURRENCY", "200")
)  # Количество параллельно обрабатываемых файлов графа

# Создание директорий если они не существуют
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_NODES_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_RELATIONS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
