"""
Применить HNSW-конфиг к коллекции Qdrant (создать или обновить существующую).

    python -m src.indexing.configure_qdrant
    python -m src.indexing.configure_qdrant --m 32 --ef-construct 200
"""

import argparse

from qdrant_client import QdrantClient

from src.indexing import config
from src.indexing.qdrant_collection import DEFAULT_HNSW, HnswParams, ensure_collection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Применить HNSW-конфиг к коллекции Qdrant")
    p.add_argument("--m", type=int, default=DEFAULT_HNSW.m,
                   help=f"рёбер на узел (default: {DEFAULT_HNSW.m})")
    p.add_argument("--ef-construct", type=int, default=DEFAULT_HNSW.ef_construct,
                   help=f"ef_construct (default: {DEFAULT_HNSW.ef_construct})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hnsw = HnswParams(m=args.m, ef_construct=args.ef_construct)

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_HTTP_PORT)
    print(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_HTTP_PORT}, "
          f"коллекция '{config.QDRANT_COLLECTION}'")
    print(f"Применяю HNSW: m={hnsw.m}, ef_construct={hnsw.ef_construct}")

    ensure_collection(client, hnsw=hnsw)

    info = client.get_collection(config.QDRANT_COLLECTION)
    print(f"Готово. Точек: {info.points_count}, статус: {info.status}")
    print("Qdrant переиндексирует коллекцию в фоне; дождись статуса 'green'.")


if __name__ == "__main__":
    main()
