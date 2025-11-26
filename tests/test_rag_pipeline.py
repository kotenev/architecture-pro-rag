#! /usr/bin/env python
import os
import pytest
from yandex_cloud_ml_sdk import YCloudML

from src.rag_bot import RAGBot
from src.config import INDEX_DIR

@pytest.fixture(scope="session")
def yandex_env():
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    api_key = os.environ.get("YANDEX_API_KEY")
    model = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-5-lite")

    if not folder_id or not api_key:
        pytest.skip("YANDEX_FOLDER_ID и YANDEX_API_KEY не заданы. Пропуск тестов.")

    return {
        "folder_id": folder_id,
        "api_key": api_key,
        "model": model,
    }


@pytest.fixture(scope="session")
def rag_bot(yandex_env):
    bot = RAGBot(index_dir=INDEX_DIR)
    assert len(bot.chunks) > 0, "Индекс пуст или не загружен"
    return bot


def test_sdk_connectivity(yandex_env):
    sdk = YCloudML(folder_id=yandex_env["folder_id"], auth=yandex_env["api_key"])

    has_chat = hasattr(sdk.models, "chat")
    has_completions = hasattr(sdk.models, "completions")

    assert has_chat or has_completions, "SDK не имеет chat или completions API"

    if has_chat:
        model_builder = sdk.models.chat
        api_mode = "chat"
    else:
        model_builder = sdk.models.completions
        api_mode = "completions"

    model_client = model_builder(yandex_env["model"]).configure(temperature=0.0, max_tokens=64)

    if api_mode == "chat":
        result = model_client.run([{"role": "user", "text": "Привет!"}])
    else:
        result = model_client.run("Привет!")

    text = None
    if hasattr(result, "result") and hasattr(result.result, "alternatives"):
        text = result.result.alternatives[0].text
    elif hasattr(result, "alternatives"):
        text = result.alternatives[0].text
    elif isinstance(result, list) and hasattr(result[0], "text"):
        text = result[0].text

    assert text is not None and len(text.strip()) > 0, "Модель не вернула текстовый ответ"


def test_rag_pipeline_basic(rag_bot):
    query = "Что делает модуль ContextManager?"
    result = rag_bot.answer(query)

    assert isinstance(result, dict)
    assert "answer" in result
    assert len(result["answer"].strip()) > 0
    assert "source" in result
    assert isinstance(result["source"], list)
    assert result["explain"] == "OK" or "Ошибка" not in result["explain"]


def test_rag_no_answer_case(rag_bot):
    query = "Какая температура на Сатурне сейчас?"
    result = rag_bot.answer(query)

    assert isinstance(result, dict)
    assert "answer" in result
    assert "я не знаю" in result["answer"].lower(), "Бот должен вернуть 'Я не знаю.' при отсутствии контекста"
    assert result["source"] == [] or len(result["source"]) == 0
