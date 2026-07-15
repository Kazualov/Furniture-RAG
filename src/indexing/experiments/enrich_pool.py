import csv
import asyncio
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.database.database import LiveDatabaseClient


async def enrich_pool():
    print("Загрузка модели (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(SCRIPT_DIR, 'relevance_template.csv')
    output_file = os.path.join(SCRIPT_DIR, '../../search/golden_set.csv')

    # 1. Читаем текущий датасет, чтобы понять, какие пары (запрос-товар) уже есть
    print(f"Чтение текущего пула из {input_file}...")
    print(input_file)
    df = pd.read_csv(input_file)

    # Собираем существующие пары в set для быстрого O(1) поиска
    existing_pairs = set(zip(df['query_id'], df['product_id']))

    # Вытаскиваем уникальные запросы, которые у нас есть (все 30 штук)
    queries = df[['query_id', 'query']].drop_duplicates().to_dict('records')

    new_rows = []

    print("Подключение к базе данных...")
    await LiveDatabaseClient.connect()

    try:
        for q in queries:
            q_id = q['query_id']
            q_text = q['query']

            # 2. Делаем векторный поиск (берем top-200, как было у лексического)
            query_vector = encoder.encode(q_text, normalize_embeddings=True).tolist()
            dense_results = await LiveDatabaseClient.search_dense(query_vector, limit=200)

            # 3. Фильтруем результаты и находим новые товары
            added_for_query = 0
            for item in dense_results:
                product_id = item.product_id

                # Если этой пары еще нет в датасете — добавляем как нового кандидата
                if (q_id, product_id) not in existing_pairs:
                    new_rows.append({
                        'query_id': q_id,
                        'query': q_text,
                        'product_id': product_id,
                        'title': item.metadata.title,
                        'description': item.metadata.description or "",
                        'relevance': ""  # Оставляем пустым для разметки LLM-судьей
                    })
                    # Добавляем в set, чтобы избежать дублей, если они вдруг всплывут
                    existing_pairs.add((q_id, product_id))
                    added_for_query += 1

            print(f"Запрос {q_id}: найдено {added_for_query} новых семантических кандидатов.")

    finally:
        await LiveDatabaseClient.disconnect()

    # 4. Сохраняем результат
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Присоединяем новые строки к старому датасету
        result_df = pd.concat([df, new_df], ignore_index=True)

        # Сохраняем с правильным квотированием строк (как в исходном файле)
        result_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"\nУспешно! Добавлено {len(new_rows)} новых кандидатов.")
        print(f"Обновленный датасет сохранен в {output_file}.")
    else:
        print("\nНовых кандидатов не найдено, все уже есть в пуле.")


if __name__ == "__main__":
    asyncio.run(enrich_pool())
