#!/usr/bin/env python3
"""Idempotent OpenSearch index initializer for local docker runs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from urllib.parse import urlparse

from langchain_openai import OpenAIEmbeddings
from opensearchpy import OpenSearch

from src.agent.vector_store.vector_store import VectorStore
from src.config import settings


def _client_from_es_url(es_url: str) -> OpenSearch:
    normalized = es_url if es_url.startswith(("http://", "https://")) else f"http://{es_url}"
    parsed = urlparse(normalized)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9200
    use_ssl = (parsed.scheme or "http") == "https"
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        use_ssl=use_ssl,
        verify_certs=False,
    )


def _expected_docs_count(pickle_path: Path) -> int:
    with pickle_path.open("rb") as f:
        docs = pickle.load(f)
    return len(docs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize OpenSearch index if needed")
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Drop and rebuild index even if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pickle_path = Path(settings.ES_PICKLE_DOCUMENTS_PATH)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle documents file not found: {pickle_path}")

    expected_count = _expected_docs_count(pickle_path)
    client = _client_from_es_url(settings.ES_URL)
    if not client.ping():
        raise RuntimeError(f"OpenSearch is not reachable at {settings.ES_URL}")

    index_name = settings.ES_INDEX_NAME
    exists = client.indices.exists(index=index_name)
    current_count = client.count(index=index_name)["count"] if exists else 0

    print(f"ES_URL={settings.ES_URL}")
    print(f"index={index_name}")
    print(f"exists={exists}")
    print(f"current_count={current_count}")
    print(f"expected_count={expected_count}")

    if exists and current_count == expected_count and not args.force_reload:
        print("Index is already initialized. Nothing to do.")
        return 0

    if exists and current_count > 0 and not args.force_reload:
        print(
            "Index exists but count differs from expected. "
            "Run with --force-reload to rebuild."
        )
        return 2

    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMB_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    VectorStore(
        embedding_function=embeddings,
        index_name=index_name,
        batch_size=settings.ES_BATCH_SIZE,
        pickle_documents_path=str(pickle_path),
        full_reload=args.force_reload,
        faiss_path=settings.FAISS_PATH,
        setup_index=True,
    )

    final_count = client.count(index=index_name)["count"]
    print(f"final_count={final_count}")
    if final_count != expected_count:
        raise RuntimeError(
            f"Initialization finished with unexpected count: {final_count} != {expected_count}"
        )
    print("Index initialization completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
