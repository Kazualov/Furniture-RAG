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

PARQUET_PATH = os.getenv("PARQUET_PATH", "./data/products.parquet")
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


def _clean_missing(value):
    """
    Аналог field_validator из ProductMetadata: превращает строковые
    'none'/'null'/'nan'/'' и float NaN в настоящий Python None,
    чтобы в БД не улетали мусорные строки вместо NULL.
    """
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in ("none", "null", "nan", ""):
            return None
        return value

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


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

    # Очистка "пустых" значений по тем же правилам, что и ProductMetadata
    text_cols = [
        "title", "description", "features", "categories",
        "details_text", "store", "image_url", "full_text",
    ]
    for col in text_cols:
        df[col] = df[col].apply(_clean_missing)

    df["price"] = pd.to_numeric(df["price"].apply(_clean_missing), errors="coerce")
    df["average_rating"] = pd.to_numeric(df["average_rating"].apply(_clean_missing), errors="coerce")
    df["rating_number"] = (
        pd.to_numeric(df["rating_number"].apply(_clean_missing), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(subset=["parent_asin", "title"])
    df = df.drop_duplicates(subset=["parent_asin"])

    return df


async def insert_rows(conn: asyncpg.Connection, rows: list[tuple]) -> None:
    await conn.executemany(INSERT_QUERY, rows)


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