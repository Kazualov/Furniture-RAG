"""
Этап 1: Загрузка исходных данных о товарах из .parquet в PostgreSQL.

Ожидаемые колонки в parquet-файле (как минимум):
    asin, parent_asin, title, description, price,
    main_category, average_rating, rating_number

Если у Азамата/исходного датасета названия колонок отличаются —
поправь маппинг в COLUMN_MAPPING ниже, остальной код менять не нужно.

Запуск:
    python -m src.database.load_postgres
"""

import asyncio
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
COLUMN_MAPPING = {
    "asin": "asin",
    "parent_asin": "parent_asin",
    "title": "title",
    "description": "description",
    "price": "price",
    "main_category": "main_category",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
}

INSERT_QUERY = """
    INSERT INTO products (
        asin, parent_asin, title, description,
        price, main_category, average_rating, rating_number
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (asin) DO UPDATE SET
        parent_asin    = EXCLUDED.parent_asin,
        title          = EXCLUDED.title,
        description    = EXCLUDED.description,
        price          = EXCLUDED.price,
        main_category  = EXCLUDED.main_category,
        average_rating = EXCLUDED.average_rating,
        rating_number  = EXCLUDED.rating_number;
"""


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns=COLUMN_MAPPING)

    required = list(COLUMN_MAPPING.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"В parquet-файле отсутствуют колонки: {missing}. "
            f"Поправь COLUMN_MAPPING в load_postgres.py."
        )

    df = df[required].copy()

    # Базовая очистка типов
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype(int)
    df["parent_asin"] = df["parent_asin"].fillna(df["asin"])
    df = df.dropna(subset=["asin", "title"])
    df = df.drop_duplicates(subset=["asin"])

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
                    "asin", "parent_asin", "title", "description",
                    "price", "main_category", "average_rating", "rating_number",
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
