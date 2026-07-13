"""
Этап 1: Загрузка обновленных мультимодальных векторов (512-dim) в Qdrant.
Читает матрицу .npy и сопоставляет её с parent_asin из parquet-метаданных.

Запуск:
    python -m src.database.load_qdrant
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "furniture_products")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 512))

# Пути к новым файлам, созданным мультимодальным generate_embeddings.py
NPY_PATH = "./embeddings/office_products_micro_embeddings_fp32.npy"
META_PATH = "./embeddings/office_products_micro_metadata.parquet"
BATCH_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP_PORT)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует, пересоздаю...")
        client.delete_collection(COLLECTION_NAME)

    # Создаем чистую коллекцию под размерность 512
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def load_vectors_from_npy_and_meta(npy_path: str, meta_path: str) -> pd.DataFrame:
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Не найдены новые файлы эмбеддингов!\n"
            f"Убедитесь, что сначала запустили новый generate_embeddings.py\n"
            f"Ожидались файлы:\n- {npy_path}\n- {meta_path}"
        )

    print(f"Читаю эмбеддинги из {npy_path} ...")
    vectors = np.load(npy_path)

    print(f"Читаю метаданные из {meta_path} ...")
    meta_df = pd.read_parquet(meta_path)

    id_col = "asin" if "asin" in meta_df.columns else "parent_asin"
    if id_col not in meta_df.columns:
        raise ValueError(f"Колонка идентификатора не найдена в {meta_path}")

    # Объединяем ключи товаров и массивы векторов
    df = pd.DataFrame({
        "point_key": meta_df[id_col].values,
        "vector": list(vectors)
    })
    return df


def upload(client: QdrantClient, df: pd.DataFrame) -> None:
    points = []
    for idx, row in df.iterrows():
        vector = list(row["vector"])
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"Размерность вектора {len(vector)} не совпадает с EMBEDDING_DIM={EMBEDDING_DIM} "
                f"для {row['point_key']}. Проверьте генерацию."
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
    # Загружаем сопоставленные 512-мерные данные
    df = load_vectors_from_npy_and_meta(NPY_PATH, META_PATH)
    print(f"Загружено в память: {len(df)} мультимодальных векторов")

    client = get_client()
    ensure_collection(client)
    upload(client, df)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Готово! Точек в коллекции '{COLLECTION_NAME}': {info.points_count}")


if __name__ == "__main__":
    main()