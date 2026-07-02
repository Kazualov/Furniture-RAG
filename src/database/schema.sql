-- Схема согласована с src/search/interfaces.py -> ProductMetadata
-- Эта таблица автоматически выполняется при первом старте контейнера postgres
-- (см. docker-compose.yml -> /docker-entrypoint-initdb.d)
--
-- В новом формате данных отдельного поля asin больше нет — ProductMetadata
-- группирует товары по parent_asin (DBResultItem.product_id маппится на
-- parent_asin), поэтому parent_asin становится первичным ключом таблицы.

CREATE TABLE IF NOT EXISTS products (
    parent_asin     TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT,
    features        TEXT,
    categories      TEXT,
    details_text    TEXT,
    price           DOUBLE PRECISION,
    average_rating  DOUBLE PRECISION,
    rating_number   INTEGER NOT NULL DEFAULT 0,
    store           TEXT,
    image_url       TEXT,
    full_text       TEXT,

    -- Векторный столбец для полнотекстового (лексического) поиска.
    -- GENERATED ALWAYS гарантирует, что он не рассинхронизируется с исходными полями.
    -- full_text обычно уже содержит конкатенацию title/description/features и т.п.,
    -- поэтому именно он взят с весом 'A'; остальные поля добавлены как страховка
    -- на случай, если full_text не заполнен.
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(full_text, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(features, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(categories, '')), 'D')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_products_search_vector
    ON products USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_products_store
    ON products (store);