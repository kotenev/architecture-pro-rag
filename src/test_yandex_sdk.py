#! /usr/bin/env python
import os
import sys
import argparse
from pprint import pprint
from yandex_cloud_ml_sdk import YCloudML

def main():
    parser = argparse.ArgumentParser(
        description="Проверка работы YandexGPT через yandex_cloud_ml_sdk"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Привет, кто ты?",
        help="Текст запроса к модели (по умолчанию: 'Привет, кто ты?')",
    )
    args = parser.parse_args()
    test_query = args.prompt

    YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
    YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
    YANDEX_LLM_MODEL = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-5-lite")

    if not YANDEX_FOLDER_ID or not YANDEX_API_KEY:
        print("Ошибка: переменные окружения YANDEX_FOLDER_ID и YANDEX_API_KEY не заданы.")
        print("Пример:")
        print("  export YANDEX_FOLDER_ID=<your-folder-id>")
        print("  export YANDEX_API_KEY=<your-api-key-or-iam-token>")
        sys.exit(1)

    print("Переменные окружения найдены:")
    print(f"   Folder ID : {YANDEX_FOLDER_ID}")
    print(f"   Model     : {YANDEX_LLM_MODEL}")

    sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)

    print("\nПроверка доступных методов SDK.models:")
    has_chat = hasattr(sdk.models, "chat")
    has_completions = hasattr(sdk.models, "completions")
    print(f" - chat:         {has_chat}")
    print(f" - completions:  {has_completions}")

    if has_chat:
        print("Используется .chat() API")
        model_builder = sdk.models.chat
        api_mode = "chat"
    elif has_completions:
        print("Используется .completions() API")
        model_builder = sdk.models.completions
        api_mode = "completions"
    else:
        print("Ни chat, ни completions не найдены — обновите SDK: pip install -U yandex-cloud-ml-sdk")
        sys.exit(1)

    model_client = model_builder(YANDEX_LLM_MODEL).configure(
        temperature=0.0,
        max_tokens=256,
    )

    print(f"\nВыполняется тестовый запрос к модели ({api_mode}):")
    print(f"{test_query}\n")

    try:
        if api_mode == "chat":
            result = model_client.run([{"role": "user", "text": test_query}])
        else:
            result = model_client.run(test_query)
    except Exception as e:
        print("Ошибка вызова модели:", e)
        sys.exit(1)

    print("Ответ получен!\n")
    print("Тип объекта результата:", type(result))
    print("Содержимое:")
    try:
        pprint(result)
    except Exception:
        print(result)

    text = None
    try:
        if hasattr(result, "result") and hasattr(result.result, "alternatives"):
            text = result.result.alternatives[0].text
        elif isinstance(result, list) and hasattr(result[0], "text"):
            text = result[0].text
        elif hasattr(result, "alternatives"):
            text = result.alternatives[0].text
        else:
            text = str(result)
    except Exception:
        text = str(result)

    print("\nИзвлечённый текст ответа:")
    print(text.strip() if text else "(пусто)")

    print("\nТест завершён успешно.")


if __name__ == "__main__":
    main()
