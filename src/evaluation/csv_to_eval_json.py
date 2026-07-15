import os
import json
import pandas as pd

# Динамическое определение путей относительно расположения скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, 'golden_set.csv')  # или golden_set.csv
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'evaluation_dataset.json')

# Порог релевантности:
# 1 — учитывать оценки 1 и 2 (мягкий порог)
# 2 — учитывать только оценку 2 (строгий порог)
RELEVANCE_THRESHOLD = 1


def convert_csv_to_json():
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден!")
        return

    print(f"Чтение разметки из {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Отфильтровываем строки, где нет разметки (NaN), и применяем порог
    # Считаем релевантными только товары с оценкой >= RELEVANCE_THRESHOLD
    relevant_df = df[df['relevance'].notna() & (df['relevance'] >= RELEVANCE_THRESHOLD)]

    # 2. Группируем по тексту запроса и собираем product_id в списки
    # (Используем list(set(...)), чтобы избежать случайных дубликатов в рамках одного запроса)
    grouped = relevant_df.groupby('query')['product_id'].apply(lambda x: list(set(x))).reset_index()

    # 3. Формируем структуру списка словарей под формат eval.py
    evaluation_dataset = []
    for _, row in grouped.iterrows():
        evaluation_dataset.append({
            "query": row['query'],
            "relevant_product_ids": row['product_id']
        })

    # 4. Записываем результат в JSON-файл
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluation_dataset, f, indent=2, ensure_ascii=False)

    print(f"\nУспешно сконвертировано!")
    print(f"Итоговый файл сохранен в: {OUTPUT_FILE}")
    print(f"Количество запросов в датасете: {len(evaluation_dataset)}")
    print(
        f"Среднее кол-во релевантных товаров на запрос: {relevant_df.groupby('query')['product_id'].count().mean():.2f}")


if __name__ == "__main__":
    convert_csv_to_json()
