-- Схема согласована с src/search/interfaces.py -> ProductMetadata
-- Эта таблица автоматически выполняется при первом старте контейнера postgres
-- (см. docker-compose.yml -> /docker-entrypoint-initdb.d)

CREATE TABLE IF NOT EXISTS products (
    asin            TEXT PRIMARY KEY,
    parent_asin     TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    price           DOUBLE PRECISION,
    main_category   TEXT,
    average_rating  DOUBLE PRECISION,
    rating_number   INTEGER NOT NULL DEFAULT 0,

    -- Векторный столбец для полнотекстового (лексического) поиска.
    -- GENERATED ALWAYS гарантирует, что он не рассинхронизируется с title/description.
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_products_search_vector
    ON products USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_products_parent_asin
    ON products (parent_asin);
