"""
Применение SQL-миграций лексического индекса к PostgreSQL.

Прогоняет по порядку все *.sql из migrations/, подставляя словарь
{{FTS_LANGUAGE}}. Миграции идемпотентны, запуск повторяем.

    python -m src.indexing.apply_postgres_migrations
"""

import asyncio
import pathlib

import asyncpg

from src.indexing import config

MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"


def load_migrations() -> list[tuple[str, str]]:
    """Возвращает [(имя_файла, sql_с_подставленным_словарём), ...] по порядку."""
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    migrations = []
    for path in files:
        sql = path.read_text(encoding="utf-8")
        sql = sql.replace("{{FTS_LANGUAGE}}", config.FTS_LANGUAGE)
        migrations.append((path.name, sql))
    return migrations


async def main() -> None:
    migrations = load_migrations()
    if not migrations:
        print("Миграции не найдены в", MIGRATIONS_DIR)
        return

    print(f"Подключаюсь к Postgres: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}"
          f"/{config.POSTGRES_DB}")
    print(f"Словарь полнотекстового поиска: '{config.FTS_LANGUAGE}'")

    conn = await asyncpg.connect(config.POSTGRES_DSN)
    try:
        for name, sql in migrations:
            print(f"-> применяю {name} ...")
            await conn.execute(sql)

        # проверка, что индекс на месте
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_products_search_vector'"
        )
        count = await conn.fetchval("SELECT count(*) FROM products")
        print(f"Готово. GIN-индекс search_vector: {'есть' if exists else 'НЕ создан'}. "
              f"Строк в products: {count}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
