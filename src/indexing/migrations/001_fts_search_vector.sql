-- Лексический индекс для полнотекстового поиска. Идемпотентно (IF NOT EXISTS),
-- запуск повторяем. Словарь {{FTS_LANGUAGE}} подставляется раннером; в
-- GENERATED-выражении имя словаря должно быть константой (требование IMMUTABLE).

-- tsvector-колонка: токенизированное представление текста. Веса setweight:
-- title='A', description='B'.
ALTER TABLE products
    ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('{{FTS_LANGUAGE}}', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('{{FTS_LANGUAGE}}', coalesce(description, '')), 'B')
    ) STORED;

-- GIN-индекс поверх tsvector: ускоряет полнотекстовый поиск (@@).
CREATE INDEX IF NOT EXISTS idx_products_search_vector
    ON products USING GIN (search_vector);
