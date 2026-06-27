"""Конфигурация коллекции Qdrant и параметров индекса HNSW (m, ef_construct)."""

from dataclasses import dataclass

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)

from src.indexing import config


@dataclass(frozen=True)
class HnswParams:
    """Параметры построения графа HNSW.

    m — рёбер на узел; ef_construct — ширина поиска при построении;
    full_scan_threshold — порог перехода на полный перебор.
    """

    m: int = 16
    ef_construct: int = 100
    full_scan_threshold: int = 10_000

    def to_diff(self) -> HnswConfigDiff:
        return HnswConfigDiff(
            m=self.m,
            ef_construct=self.ef_construct,
            full_scan_threshold=self.full_scan_threshold,
        )


# Базовая конфигурация HNSW.
DEFAULT_HNSW = HnswParams(m=16, ef_construct=100)


def vectors_config() -> VectorParams:
    """Конфиг вектора: размерность + косинус."""
    return VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE)


def vectors_config_flat() -> VectorParams:
    """Вектор с отключённым HNSW (m=0) — режим Flat/brute-force для сравнения."""
    return VectorParams(
        size=config.EMBEDDING_DIM,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(m=0),
    )


def ensure_collection(client, hnsw: HnswParams = DEFAULT_HNSW,
                      collection_name: str | None = None) -> None:
    """Создаёт коллекцию с HNSW-конфигом либо обновляет существующую.

    Данные не удаляются: для существующей коллекции параметры применяются
    через update_collection с последующей переиндексацией.
    """
    name = collection_name or config.QDRANT_COLLECTION
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        client.update_collection(
            collection_name=name,
            hnsw_config=hnsw.to_diff(),
        )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config(),
            hnsw_config=hnsw.to_diff(),
            optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        )
