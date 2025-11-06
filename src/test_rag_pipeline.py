#!/usr/bin/env python
import os
import sys
from pprint import pprint

from src.rag_bot import RAGBot
from src.config import KB_DIR, INDEX_DIR, FEWSHOT_FILE

YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_LLM_MODEL = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-5-lite")

if not YANDEX_FOLDER_ID or not YANDEX_API_KEY:
    print("Не заданы YANDEX_FOLDER_ID и/или YANDEX_API_KEY (см. .env).")
    sys.exit(1)

print("✅ Проверка окружения:")
print(f"   Folder ID : {YANDEX_FOLDER_ID}")
print(f"   Model     : {YANDEX_LLM_MODEL}")
print(f"   KB_DIR    : {KB_DIR}")
print(f"   INDEX_DIR : {INDEX_DIR}")

try:
    bot = RAGBot(index_dir=INDEX_DIR)
except Exception as e:
    print(f"Ошибка при инициализации RAGBot: {e}")
    sys.exit(1)

print("\nБот успешно инициализирован.")

fewshot_examples = []
if FEWSHOT_FILE and os.path.exists(FEWSHOT_FILE):
    import json
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fewshot_examples.append(json.loads(line))
            except Exception:
                pass

if fewshot_examples:
    print(f"Загружено few-shot примеров: {len(fewshot_examples)}")
else:
    print("Few-shot примеры не найдены — будет использоваться только контекст базы.")

user_query = input("\nВведите тестовый запрос: ").strip()
if not user_query:
    user_query = "Как Якуб заботится о понравившейся ему лошади?"

print(f"\nВыполняется полный RAG-пайплайн для запроса: '{user_query}'\n")

try:
    result = bot.answer(user_query, fewshot_examples=fewshot_examples)
except Exception as e:
    print(f"Ошибка при выполнении пайплайна: {e}")
    sys.exit(1)

print("Ответ получен!\n")
pprint(result)

print("\nИтоговый ответ:")
print(result.get("answer", "(нет ответа)"))

print("\nИспользованные источники:")
for s in result.get("source", []):
    print(" -", s)

print("\nПояснение:", result.get("explain", ""))
print("\nТест завершён успешно.")
