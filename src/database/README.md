# Этап 1 — Инфраструктура БД

## Что готово

- `docker-compose.yml` — поднимает PostgreSQL 16 и Qdrant.
- `src/database/schema.sql` — таблица `products`, схема согласована с
  `ProductMetadata` из `src/search/interfaces.py` (поля Алекса). Применяется
  автоматически при первом старте контейнера postgres.
- `src/database/load_postgres.py` — грузит `.parquet` с товарами в Postgres.
- `src/database/load_qdrant.py` — грузит вектора (от Азамата) в Qdrant.

## Как запустить

1. Скопировать `.env.example` -> `.env`, поправить пути `PARQUET_PATH` /
   `VECTORS_PATH` и при необходимости `EMBEDDING_DIM` (размерность вектора
   модели, которую использует Азамат).

2. Поднять БД:
   ```bash
   docker compose up -d
   ```
   Проверить, что обе поднялись:
   ```bash
   docker compose ps
   ```

3. Установить зависимости:
   ```bash
   pip install -r src/database/requirements.txt
   ```

4. Залить данные:
   ```bash
   python -m src.database.load_postgres
   python -m src.database.load_qdrant
   ```

## На что обратить внимание

- **`EMBEDDING_DIM`** — должен совпадать с моделью эмбеддингов Азамата
  (например, 384 для multilingual-e5-small, 768 для multilingual-e5-base и т.д.)
  Уточни у него до запуска `load_qdrant.py`.
- **Язык полнотекстового поиска** в `schema.sql` — сейчас `'english'`. Если
  названия/описания товаров на русском, замени конфигурацию словаря на
  `'russian'` (или `'simple'`, если данные смешанные) в `to_tsvector(...)`.
- **Связка Postgres <-> Qdrant**: точка в Qdrant хранит `product_id` в payload
  (равен `asin`), а в Postgres ключ — тоже `asin`. Это нужно для Этапа 2,
  чтобы можно было дотягивать полную карточку товара из Postgres по
  результату векторного поиска, и наоборот.
- Скрипты идемпотентны: `load_postgres.py` делает `UPSERT` по `asin`,
  `load_qdrant.py` пересоздаёт коллекцию с нуля — это нормально для
  повторных запусков на этапе разработки.

## Дальше — Этап 2

Обёртки `search_sparse()` / `search_dense()`, сигнатурно совпадающие с
`MockDBClient` из `database_mock.py`, чтобы Алекс просто заменил импорт
мока на реальный клиент без правок в своём коде.
