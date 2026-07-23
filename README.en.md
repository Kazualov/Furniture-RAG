# Furniture-RAG-CLIP — hybrid search engine for a product catalog

Hybrid (sparse + dense) search over a product catalog: full-text search in
PostgreSQL is combined with vector search in Qdrant, the results are merged with
**Weighted Reciprocal Rank Fusion (RRF)**, and everything is served through a
single FastAPI REST endpoint. Embeddings are produced by an in-house pipeline
(MiniLM → INT8 → multimodal CLIP).

---

## Table of contents

- [What it is and why](#what-it-is-and-why)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [API](#api)

---

## What it is and why

Classic full-text search is good at exact word matches ("office chair") but does
not understand meaning. Vector search understands meaning ("a chair for work"
≈ "office chair") but struggles with rare terms, SKUs, and brands. This project
combines both approaches:

- **Sparse (lexical)** — PostgreSQL FTS with a `tsvector` + GIN index, ranked by `ts_rank_cd`.
- **Dense (semantic)** — Qdrant, HNSW index, cosine similarity.
- **Fusion** — weighted RRF with an `alpha` parameter that tunes the balance between exact and semantic search on the fly.

The dataset is product cards (Amazon-like format: `parent_asin`, `title`,
`description`, `features`, `price`, `rating`, etc.).

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              FastAPI  /search                │
                    │           (src/search/main.py)               │
                    │                                              │
   query ──────────▶│  encode(query) ──▶ all-MiniLM-L6-v2 (384d)   │
                    │        │                                     │
                    │        ├──────────────┬──────────────┐       │
                    │        ▼              ▼              │       │
                    │  search_sparse   search_dense        │       │
                    └────────┼──────────────┼──────────────┼───────┘
                             ▼              ▼              ▼
                    ┌────────────────┐ ┌──────────┐  Weighted RRF
                    │  PostgreSQL    │ │  Qdrant  │  (alpha balance)
                    │  FTS + GIN     │ │  HNSW    │        │
                    │  (cards +      │ │ (vectors)│        ▼
                    │   search_vector)│ └──────────┘   ranked list
                    └────────────────┘      │          of products
                             ▲               │
                             └── hydration ──┘
                        (by product_id = parent_asin)
```

The key link: a point in Qdrant stores `product_id` (equal to `parent_asin`) in
its payload, while the full product card always lives in PostgreSQL. After the
vector search, the metadata is "hydrated" from Postgres by this key — vectors do
not duplicate the cards, and there is a single source of truth.

---

## Repository layout

```
src/
├── models/     Embeddings: MiniLM (FP32) · INT8 quantization · CLIP · interfaces.py
├── database/   docker-compose · schema.sql · load_postgres/qdrant · LiveDatabaseClient
├── indexing/   config · FTS migrations · HNSW config · benchmark · smoke_test
└── search/     FastAPI /search (weighted RRF) · test_api.py
```

Details for each module are in its own README.

---

## How it works

1. **Indexing (offline).** Product cards are loaded into PostgreSQL; for each row
   the `search_vector` is computed automatically (GENERATED ALWAYS — it cannot
   drift out of sync with the source fields). Embeddings are produced by the
   `models/` pipeline and loaded into Qdrant with `product_id` in the payload.

2. **Query (online).** FastAPI encodes the query text into a vector
   (all-MiniLM-L6-v2, 384d, L2-normalized) and runs both searches in parallel
   (`asyncio.gather`):
   - `search_sparse` — FTS over `search_vector` via `websearch_to_tsquery`, ranked by `ts_rank_cd`;
   - `search_dense` — nearest-vector search in Qdrant + metadata hydration from Postgres.

3. **Fusion.** The two ranked lists are merged with weighted RRF:

   ```
   score(doc) = alpha · Σ 1/(k + rank_sparse)  +  (1 - alpha) · Σ 1/(k + rank_dense)
   ```

   where `k = 60` (the RRF standard) and `alpha ∈ [0, 1]` is set per request:
   `alpha = 0` — pure semantics, `alpha = 1` — pure lexical search. Empirically,
   the benchmarks give an optimum of **`alpha = 0.6`**.

### FTS weighting scheme (`schema.sql`)

The full-text vector is built from several fields with different weights:

| Field         | Weight | Rationale                                  |
|---------------|--------|--------------------------------------------|
| `full_text`   | A      | Ready-made concatenation — the main signal |
| `title`       | A      | Title — the most relevant feature          |
| `description` | B      | Description                                |
| `features`    | C      | Feature bullet points                      |
| `categories`  | D      | Categories — a background signal           |

---

## Quick start

### 0. Requirements

- Python 3.11+
- Docker + Docker Compose
- ~2 GB RAM for the database containers (more for the full dataset)

### 1. Start the databases

```bash
cd src/database
cp env.example ../../.env        # then fix the data paths
docker compose up -d
docker compose ps                # both services should be healthy
```

`schema.sql` is applied automatically on the first start of the postgres container.

### 2. Install dependencies

```bash
pip install -r src/database/requirements.txt
pip install -r src/indexing/requirements.txt
pip install -r src/models/requirements.txt   # if you run the embedding pipeline
# additionally for the API:
pip install fastapi uvicorn sentence-transformers
```

### 3. Compute embeddings (if you don't have vectors yet)

```bash
cd src/models
python generate_embeddings.py --input ../../office_products_micro.parquet
python quantize.py            --input ../../office_products_micro.parquet   # optional, INT8
python multimodal.py          --input ../../office_products_micro.parquet   # optional, CLIP
```

Details, output file formats, and how to work with the INT8 scale are in
[`models/README.md`](models/README.md).

### 4. Load the data

```bash
python -m src.database.load_postgres    # cards → Postgres
python -m src.database.load_qdrant      # vectors → Qdrant
```

Both scripts are idempotent: `load_postgres` does an UPSERT by `parent_asin`,
`load_qdrant` recreates the collection from scratch.

### 5. Configure the indexes

```bash
python -m src.indexing.apply_postgres_migrations   # tsvector + GIN
python -m src.indexing.configure_qdrant            # HNSW config
python -m src.indexing.smoke_test                  # check both indexes
```

### 6. Run the API

```bash
python -m src.search.main
# → http://localhost:8000  (Swagger UI: http://localhost:8000/docs)
```

## API

### `POST /search`

**Request:**

```json
{
  "query": "black leather office chair",
  "limit": 10,
  "alpha": 0.6
}
```

| Field   | Type   | Default              | Description                                                  |
|---------|--------|----------------------|--------------------------------------------------------------|
| `query` | string | —                    | Search query text                                            |
| `limit` | int    | `10`                 | How many results to return                                   |
| `alpha` | float  | `0.5` (rec. `0.6`)   | Sparse/dense balance: `0.0` = semantics, `1.0` = lexical. Benchmarks give an optimum of `0.6` |

**Response** — a list of products sorted by `rrf_score`:

```json
[
  {
    "product_id": "B00YQ6X8EO",
    "rrf_score": 0.016393,
    "metadata": {
      "parent_asin": "B00YQ6X8EO",
      "title": "...",
      "price": 129.99,
      "average_rating": 4.5,
      "image_url": "https://...",
      "...": "..."
    }
  }
]
```

Example with `curl`:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "black leather office chair", "limit": 5, "alpha": 0.6}'
```

Interactive docs — `http://localhost:8000/docs`.
