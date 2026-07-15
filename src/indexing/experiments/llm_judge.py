import os
import csv
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Подключаемся к локальному серверу Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ключ не нужен, но библиотека требует передать хоть что-то
)

# Выбираем скачанную модель
MODEL_NAME = "llama3.1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, '../../search/golden_set.csv')

SYSTEM_PROMPT = """
Ты — строгий эксперт по оценке релевантности поиска.
Оцени релевантность ТОВАРА для ЗАПРОСА по шкале от 0 до 2:
2 = Точное попадание. Товар идеально решает задачу.
1 = Частичное совпадение. Смежная категория, аксессуар или не совсем точная модель.
0 = Мимо. Товар не подходит под запрос.

ВЫВЕДИ ТОЛЬКО ОДНУ ЦИФРУ (0, 1 или 2). Никаких слов, пояснений и точек.
"""


def get_relevance_score(query: str, title: str, description: str) -> int:
    """Обращается к Ollama и безопасно извлекает цифру."""
    user_prompt = f"ЗАПРОС: {query}\n\nТОВАР:\nНазвание: {title}\nОписание: {description}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Максимальная предсказуемость
            max_tokens=10  # Разрешаем выдать чуть больше на случай, если модель начнет "болтать"
        )

        result_text = response.choices[0].message.content.strip()

        # Регулярное выражение: ищем первую попавшуюся цифру 0, 1 или 2 в ответе модели
        match = re.search(r'[012]', result_text)
        if match:
            return int(match.group(0))
        else:
            print(f"\n[Предупреждение] Модель вернула текст без нужных цифр: '{result_text}'")
            return None

    except Exception as e:
        print(f"\nОшибка при обращении к Ollama: {e}")
        print("Убедись, что Ollama запущена (ollama run llama3.1)")
        return None


def run_judge():
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден!")
        return

    print(f"Читаем файл {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    missing_mask = df['relevance'].isna() | (df['relevance'] == "")
    indices_to_score = df[missing_mask].index.tolist()

    if not indices_to_score:
        print("Всё уже размечено!")
        return

    print(f"Найдено {len(indices_to_score)} пар без оценки. Модель: {MODEL_NAME}")

    # Итерируемся по нужным строкам
    for idx in tqdm(indices_to_score, desc="Оценка Ollama"):
        row = df.loc[idx]
        query = row['query']
        title = row['title']
        description = row['description'] if pd.notna(row['description']) else ""

        score = get_relevance_score(query, title, description)

        if score is not None:
            df.at[idx, 'relevance'] = score

            # Локальные модели бесплатные, поэтому сохраняем реже, например каждые 20 строк
            if idx % 20 == 0:
                df.to_csv(INPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            # Если модель сбоит, просто пропускаем и идем дальше, потом можно будет доразметить
            pass

    df.to_csv(INPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("\nРазметка успешно завершена!")


if __name__ == "__main__":
    run_judge()
