"""
Этап 1: Загрузка исходных данных о товарах из .parquet в PostgreSQL.

Ожидаемые колонки в parquet-файле (как минимум):
    parent_asin, title, description, features, categories,
    details_text, price, average_rating, rating_number,
    store, image_url, full_text

Запуск:
    python -m src.database.load_postgres
"""

import asyncio
import math
import os

import asyncpg
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

POSTGRES_DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'rag_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'rag_password')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', 5432)}/"
    f"{os.getenv('POSTGRES_DB', 'furniture_db')}"
)

PARQUET_PATH = os.getenv("PARQUET_PATH", "./embeddings/office_products_micro.parquet")
BATCH_SIZE = 1000

# Если колонки в исходном файле называются иначе — меняем только здесь.
# Порядок значений (правая часть) ничего не определяет, главное —
# чтобы все поля ProductMetadata присутствовали в итоговом df.
COLUMN_MAPPING = {
    "parent_asin": "parent_asin",
    "title": "title",
    "description": "description",
    "features": "features",
    "categories": "categories",
    "details_text": "details_text",
    "price": "price",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "store": "store",
    "image_url": "image_url",
    "full_text": "full_text",
}

REQUIRED_COLUMNS = list(COLUMN_MAPPING.values())

INSERT_QUERY = """
    INSERT INTO products (
        parent_asin, title, description, features, categories,
        details_text, price, average_rating, rating_number,
        store, image_url, full_text
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ON CONFLICT (parent_asin) DO UPDATE SET
        title          = EXCLUDED.title,
        description    = EXCLUDED.description,
        features       = EXCLUDED.features,
        categories     = EXCLUDED.categories,
        details_text   = EXCLUDED.details_text,
        price          = EXCLUDED.price,
        average_rating = EXCLUDED.average_rating,
        rating_number  = EXCLUDED.rating_number,
        store          = EXCLUDED.store,
        image_url      = EXCLUDED.image_url,
        full_text      = EXCLUDED.full_text;
"""


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_text(value) -> str | None:
    """Normalize text columns to str | None for asyncpg TEXT fields."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "null", "nan"):
            return None
        return cleaned
    return str(value)


def _clean_float(value) -> float | None:
    """Normalize nullable numeric columns for asyncpg DOUBLE PRECISION fields."""
    if _is_missing(value):
        return None
    return float(value)


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns=COLUMN_MAPPING)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В parquet-файле отсутствуют колонки: {missing}. "
            f"Поправь COLUMN_MAPPING в load_postgres.py."
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Очистка "пустых" значений по тем же правилам, что и ProductMetadata.
    # Явно приводим текстовые колонки к object + None: itertuples иначе
    # может отдать float NaN вместо NULL для asyncpg.
    text_cols = [
        "parent_asin", "title", "description", "features", "categories",
        "details_text", "store", "image_url", "full_text",
    ]
    for col in text_cols:
        df[col] = df[col].map(_clean_text).astype(object)

    df["price"] = pd.to_numeric(df["price"].map(_clean_text), errors="coerce").map(_clean_float)
    df["average_rating"] = pd.to_numeric(
        df["average_rating"].map(_clean_text), errors="coerce"
    ).map(_clean_float)
    df["rating_number"] = (
        pd.to_numeric(df["rating_number"].map(_clean_text), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(subset=["parent_asin", "title"])
    df = df.drop_duplicates(subset=["parent_asin"])

    return df


def _sanitize_row(row: tuple) -> tuple:
    """Last-line guard: asyncpg TEXT params must be str | None, never float NaN."""
    (
        parent_asin, title, description, features, categories,
        details_text, price, average_rating, rating_number,
        store, image_url, full_text,
    ) = row
    return (
        _clean_text(parent_asin),
        _clean_text(title),
        _clean_text(description),
        _clean_text(features),
        _clean_text(categories),
        _clean_text(details_text),
        _clean_float(price),
        _clean_float(average_rating),
        int(rating_number) if not _is_missing(rating_number) else 0,
        _clean_text(store),
        _clean_text(image_url),
        _clean_text(full_text),
    )


async def insert_rows(conn: asyncpg.Connection, rows: list[tuple]) -> None:
    await conn.executemany(INSERT_QUERY, [_sanitize_row(r) for r in rows])


async def main() -> None:
    print(f"Читаю {PARQUET_PATH} ...")
    df = load_dataframe(PARQUET_PATH)
    print(f"Загружено в память: {len(df)} строк")

    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        records = list(
            df[
                [
                    "parent_asin", "title", "description", "features",
                    "categories", "details_text", "price", "average_rating",
                    "rating_number", "store", "image_url", "full_text",
                ]
            ].itertuples(index=False, name=None)
        )

        for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Заливка в Postgres"):
            batch = records[i : i + BATCH_SIZE]
            await insert_rows(conn, batch)

        count = await conn.fetchval("SELECT COUNT(*) FROM products;")
        print(f"Готово. Всего строк в таблице products: {count}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())