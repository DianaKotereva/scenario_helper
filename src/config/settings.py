import os

import yaml
from dotenv import load_dotenv

load_dotenv()

# OUTPUT_FILE_SAVE_PATH: Path = Path(os.getenv("OUTPUT_FILE_SAVE_PATH"))

BM25_K: int = int(os.getenv("BM25_K"))
ES_RETRIEVER_K: int = int(os.getenv("ES_RETRIEVER_K"))
ES_SIMILARITY_THRESHOLD: float = float(os.getenv("ES_SIMILARITY_THRESHOLD"))

ES_URL: str = os.getenv("ES_URL")
ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME")
ES_BATCH_SIZE: int = int(os.getenv("ES_BATCH_SIZE"))
ES_PICKLE_DOCUMENTS_PATH: str = os.getenv("ES_PICKLE_DOCUMENTS_PATH")

GRAPH_PICKLE_PATH: str = os.getenv("GRAPH_PICKLE_PATH")

# PICKLE_GRAPHS_PATH: Path = Path(os.getenv("PICKLE_GRAPHS_PATH"))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE")
OPENAI_API_MODEL: str = os.getenv("OPENAI_API_MODEL")
OPENAI_EMB_MODEL: str = os.getenv("OPENAI_EMB_MODEL")

LLM_GIGACHAT_BASE_URL: str = os.getenv("LLM_GIGACHAT_BASE_URL")
LLM_GIGACHAT_MODEL_NAME: str = os.getenv("LLM_GIGACHAT_MODEL_NAME")
LLM_GIGACHAT_API_KEY: str = os.getenv("LLM_GIGACHAT_API_KEY")
LLM_GIGACHAT_TEMPERATURE: float = float(os.getenv("LLM_GIGACHAT_TEMPERATURE"))
LLM_GIGACHAT_SCOPE: str = os.getenv("LLM_GIGACHAT_SCOPE")

LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECt: str = os.getenv("LANGSMITH_PROJECt")

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")

TOP_K_GRAPHS: int = int(os.getenv("TOP_K_GRAPHS"))
LLM_TYPE: str = os.getenv("LLM_TYPE")

MAX_N_ITERATIONS: int = int(os.getenv("MAX_N_ITERATIONS"))

USE_FAISS: bool = os.getenv("USE_FAISS", "False") == "True"
FAISS_PATH: str = os.getenv("FAISS_PATH")
# SPLIT_QUESTIONS: bool = os.getenv("SPLIT_QUESTIONS") == "True"

with open("src/config/open_search_settings.yaml", "r", encoding="utf-8") as f:
    open_search_settings = yaml.safe_load(f)
