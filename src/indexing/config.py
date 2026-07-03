"""Конфигурация модуля indexing: читает настройки БД из переменных окружения."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- PostgreSQL ----
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "furniture_db")

POSTGRES_DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Словарь для to_tsvector: 'english' / 'russian' / 'simple'.
FTS_LANGUAGE = os.getenv("FTS_LANGUAGE", "english")

# ---- Qdrant ----
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "furniture_products")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# Путь к векторам (.parquet или .npy).
VECTORS_PATH = os.getenv("VECTORS_PATH", "./data/vectors.parquet")
