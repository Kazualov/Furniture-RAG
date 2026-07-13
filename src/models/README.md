# src/models — Embedding Generation Pipeline

Превращает `.parquet` файлы с текстовыми описаниями товаров в векторы для RAG-поиска.

---

## Быстрый старт

```bash
pip install -r requirements.txt

# Этап 1 — FP32 эмбеддинги
python stage1_generate_embeddings.py --input ../../office_products_micro.parquet

# Этап 2 — INT8 квантизация + бенчмарк
python stage2_quantize.py --input ../../office_products_micro.parquet

# Этап 3 — мультимодальные CLIP эмбеддинги (опционально)
python stage3_multimodal.py --input ../../office_products_micro.parquet --max-images 500
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

### Этап 2 — `stage2_quantize.py`

Применяет **INT8 квантизацию** (dynamic по умолчанию, или static).

**Что делает:**
1. Загружает FP32 модель
2. Применяет квантизацию (dynamic — веса, static — веса + активации)
3. Запускает бенчмарк на 500 текстах: FP32 vs INT8
4. Кодирует полный датасет INT8 моделью
5. Сохраняет результаты и CSV с метриками

**Выходные файлы:**
| Файл | Описание |
|------|----------|
| `embeddings/{stem}_embeddings_int8_dynamic.npy` | Векторы `(N, 384)`, dtype **int8** |
| `embeddings/{stem}_embeddings_int8_dynamic_scale.npy` | Масштаб `(N,)` float32 — **обязателен** для восстановления float-векторов |
| `embeddings/{stem}_benchmark_dynamic.csv` | speedup, cosine similarity |

> **Важно:** сами эмбеддинги (не только веса модели) сохраняются как `int8` для экономии диска (~4× меньше, чем FP32). Масштаб считается **отдельно для каждого вектора** по его реальному max-abs — так используется весь диапазон `[-128, 127]`, а не малая его часть (при 384 измерениях типичная компонента L2-нормированного вектора ~0.05, а не ~1.0). Без файла `_scale.npy` раскодировать `.npy` с векторами в исходные float-значения невозможно — храните и используйте оба файла вместе.

**Флаги:**
```
--quant-type   dynamic или static  (default: dynamic)
--batch-size   размер батча        (default: 64)
```

> **Почему dynamic лучше для старта:**  
> Static требует калибровочных данных и нестабильна на некоторых версиях PyTorch с трансформерами. Dynamic даёт 1.3–2× ускорение без конфигурации.

---

### Этап 3 — `stage3_multimodal.py`

Мультимодальные эмбеддинги через **CLIP ViT-B/32**.

**Стратегия fusion:**
- Строки с изображением: `alpha * text_emb + (1-alpha) * image_emb` (re-normalize)
- Строки без изображения: только text_emb

**Выходные файлы:**
| Файл | Описание |
|------|----------|
| `embeddings/{stem}_embeddings_clip_text.npy` | CLIP text-only (512-dim) |
| `embeddings/{stem}_embeddings_clip_multimodal.npy` | Fused (512-dim) |
| `embeddings/{stem}_clip_metadata.parquet` | Метаданные + has_image флаг |

**Флаги:**
```
--max-images   лимит загружаемых изображений (для тестов)
--alpha        вес текста при fusion (0.5 = равновесие)
```

---

## Структура выходных файлов

```
embeddings/
├── office_products_micro_embeddings_fp32.npy          ← Stage 1
├── office_products_micro_metadata.parquet             ← Stage 1
├── office_products_micro_embeddings_int8_dynamic.npy  ← Stage 2
├── office_products_micro_benchmark_dynamic.csv        ← Stage 2
├── office_products_micro_embeddings_clip_text.npy     ← Stage 3
├── office_products_micro_embeddings_clip_multimodal.npy
└── office_products_micro_clip_metadata.parquet
```

## Загрузка векторов (пример для команды БД/поиска)

### Вариант A — FP32 (Stage 1), самый простой

```python
import numpy as np
import pandas as pd

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

### Вариант B — INT8 (Stage 2), нужен dequantize перед поиском

```python
import numpy as np
import pandas as pd

int8_vectors = np.load("embeddings/office_products_micro_embeddings_int8_dynamic.npy")   # dtype: int8
scale        = np.load("embeddings/office_products_micro_embeddings_int8_dynamic_scale.npy")  # dtype: float32, shape (N,)
metadata     = pd.read_parquet("embeddings/office_products_micro_metadata.parquet")

# ВАЖНО: сначала переводим в float32, потом делим на scale.
# Прямое умножение/сложение int8-массивов может тихо переполниться —
# NumPy не расширяет тип аккумулятора автоматически.
embeddings = int8_vectors.astype(np.float32) / scale[:, None]   # (N, 384) float32, снова L2-нормированные

query_vec = embeddings[0]
scores = embeddings @ query_vec
top_k = scores.argsort()[::-1][:10]
print(metadata.iloc[top_k][["title", "price"]])
```

> Если БД/поисковый движок поддерживает нативное хранение INT8 (например, для экономии RAM в индексе), можно хранить `int8_vectors` как есть, но **scale обязательно храните рядом** — без него значения не восстановить в исходном масштабе.

---

## Заметки по производительности

| Конфиг | Скорость (ожидаемая) |
|--------|----------------------|
| CPU, FP32, batch=128 | ~200–400 texts/s |
| CPU, INT8 dynamic, batch=64 | ~300–700 texts/s |
| GPU (T4), FP32, batch=256 | ~2000–4000 texts/s |

На micro (5 000 строк) Stage 1 займёт ~15–30 сек на CPU.  
На full (~100k строк) — несколько минут на CPU, ~1 мин на GPU.
