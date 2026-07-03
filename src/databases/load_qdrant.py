"""
Этап 1: Загрузка векторов (от Азамата) в Qdrant.

Ожидается, что вектора лежат в parquet-файле с колонками:
    asin (или parent_asin) — ключ для связи с Postgres
    vector — list[float] фиксированной размерности (EMBEDDING_DIM)

Если Азамат отдаёт .npy + отдельный список id — см. функцию
load_vectors_from_npy() ниже и переключи вызов в main().

Запуск:
    python -m src.database.load_qdrant
"""

import os

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "furniture_products")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))
VECTORS_PATH = os.getenv("VECTORS_PATH", "./data/vectors.parquet")
BATCH_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует, пересоздаю...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def load_vectors_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    id_col = "asin" if "asin" in df.columns else "parent_asin"
    if id_col not in df.columns or "vector" not in df.columns:
        raise ValueError(
            "Ожидаю колонки 'asin'/'parent_asin' и 'vector' в файле с векторами."
        )
    return df.rename(columns={id_col: "point_key"})[["point_key", "vector"]]


def upload(client: QdrantClient, df: pd.DataFrame) -> None:
    points = []
    for idx, row in df.iterrows():
        vector = list(row["vector"])
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Размерность вектора {len(vector)} не совпадает с EMBEDDING_DIM={EMBEDDING_DIM} "
                f"для {row['point_key']}. Проверь .env."
            )
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={"product_id": row["point_key"]},
            )
        )

    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Заливка в Qdrant"):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


def main() -> None:
    print(f"Читаю вектора из {VECTORS_PATH} ...")
    df = load_vectors_from_parquet(VECTORS_PATH)
    print(f"Загружено в память: {len(df)} векторов")

    client = get_client()
    ensure_collection(client)
    upload(client, df)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Готово. Точек в коллекции '{COLLECTION_NAME}': {info.points_count}")


if __name__ == "__main__":
    main()
