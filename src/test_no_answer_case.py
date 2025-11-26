#! /usr/bin/env python
import os
import sys
from pprint import pprint

from src.rag_bot import RAGBot
from src.config import INDEX_DIR

YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_LLM_MODEL = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-5-lite")

if not YANDEX_FOLDER_ID or not YANDEX_API_KEY:
    print("Ошибка: YANDEX_FOLDER_ID или YANDEX_API_KEY не заданы.")
    sys.exit(1)

print("Переменные окружения найдены")
print(f"   Folder ID : {YANDEX_FOLDER_ID}")
print(f"   Model     : {YANDEX_LLM_MODEL}")

try:
    bot = RAGBot(index_dir=INDEX_DIR)
except Exception as e:
    print("Ошибка при инициализации бота:", e)
    sys.exit(1)

print("Бот успешно инициализирован.\n")

query = "Какая температура на Сатурне в данный момент?"
print(f"Тестовый запрос: {query}\n")

try:
    result = bot.answer(query)
except Exception as e:
    print("Ошибка выполнения запроса:", e)
    sys.exit(1)

print("Ответ получен!")
pprint(result)

print("\nИтоговый ответ:")
print(result.get("answer", "(нет ответа)"))

expected = "я не знаю"
answer_text = result.get("answer", "").lower()

if expected in answer_text:
    print("\nТест пройден: бот корректно ответил 'Я не знаю.' при отсутствии релевантных данных.")
else:
    print("\nТест не пройден: бот не вернул стандартный ответ при отсутствии контекста.")
