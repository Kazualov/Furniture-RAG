# src/models — Embedding Generation Pipeline

Превращает `.parquet` файлы с текстовыми описаниями товаров в векторы для RAG-поиска.

---

## Быстрый старт

```bash
pip install -r requirements.txt

# Этап 1 — FP32 эмбеддинги
python stage1_generate_embeddings.py --input ../../office_products_micro.parquet
```

Все результаты сохраняются в папку `embeddings/`.

---

## Этапы

### Этап 1 — `stage1_generate_embeddings.py`

Модель: `all-MiniLM-L6-v2` (sentence-transformers)  
Размерность вектора: **384**  
Нормализация: L2 → косинусное сходство == скалярное произведение

**Выходные файлы:**
| Файл | Описание |
|------|----------|
| `embeddings/{stem}_embeddings_fp32.npy` | Матрица `(N, 384)` float32 |
| `embeddings/{stem}_metadata.parquet` | Лёгкие мета-колонки для отображения |

**Флаги:**
```
--input        путь к .parquet  (default: office_products_micro.parquet)
--output-dir   куда сохранять   (default: embeddings/)
--batch-size   размер батча     (default: 128, уменьши при нехватке RAM)
--model        имя модели HF    (default: all-MiniLM-L6-v2)
```

---

## Структура выходных файлов

```
embeddings/
├── office_products_micro_embeddings_fp32.npy          ← Stage 1
├── office_products_micro_metadata.parquet             ← Stage 1
```

## Загрузка векторов (пример для команды БД/поиска)

```python
import numpy as np
import pandas as pd

# Загрузить FP32 векторы
embeddings = np.load("embeddings/office_products_micro_embeddings_fp32.npy")
metadata = pd.read_parquet("embeddings/office_products_micro_metadata.parquet")

# embeddings[i] соответствует metadata.iloc[i]
print(embeddings.shape)   # (5000, 384)
print(metadata.columns)   # parent_asin, title, price, ...

# Поиск: косинусное сходство (векторы уже L2-нормированы)
query_vec = embeddings[0]                             # пример
scores = embeddings @ query_vec                       # (N,) dot product = cosine
top_k = scores.argsort()[::-1][:10]                  # топ-10
print(metadata.iloc[top_k][["title", "price"]])
```

---
