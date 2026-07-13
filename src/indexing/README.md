# src/indexing — Индексация и оптимизация БД

Конфигурации индексов поверх инфраструктуры БД: лексический индекс в PostgreSQL
и векторный индекс HNSW в Qdrant, плюс бенчмарк для подбора параметров.

Модуль не создаёт таблицы и не заливает данные — только настраивает индексы и
использует общий `.env`.

## Структура

```
src/indexing/
├── config.py                       # чтение .env: DSN Postgres, параметры Qdrant
├── migrations/
│   └── 001_fts_search_vector.sql   # tsvector + словарь + GIN (идемпотентно)
├── apply_postgres_migrations.py    # применяет миграции через asyncpg
├── search_postgres.py              # лексический поиск (ts_rank_cd)
├── qdrant_collection.py            # конфиг HNSW + create/update коллекции
├── configure_qdrant.py             # применяет HNSW-конфиг к коллекции
├── experiments/
│   └── hnsw_benchmark.py           # перебор m/ef_construct, build time, память, HNSW vs Flat
└── smoke_test.py                   # быстрая проверка обоих индексов
```

## Запуск

```bash
pip install -r src/indexing/requirements.txt

# лексический индекс в Postgres
python -m src.indexing.apply_postgres_migrations

# HNSW-конфиг коллекции Qdrant
python -m src.indexing.configure_qdrant

# эксперименты с HNSW
python -m src.indexing.experiments.hnsw_benchmark --vectors ./data/vectors.npy
```

## Заметки

- **Словарь FTS** задаётся через `FTS_LANGUAGE` (`english` / `russian` / `simple`).
- **Ранжирование** лексического поиска — `ts_rank_cd` (TF-IDF-подобное).
- **Параметры коллекции Qdrant** (включая `hnsw_config`) задаются в
  `qdrant_collection.py` и применяются через `update_collection`, поэтому
  работают поверх уже существующей коллекции.

## Бенчмарк

`experiments/hnsw_benchmark.py` для сетки `(m, ef_construct)` меряет время
построения индекса, потребление памяти, латентность и `recall@k` относительно
точного (Flat) поиска. Результаты — в `experiments/results/` (CSV + markdown).
