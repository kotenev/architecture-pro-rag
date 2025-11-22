#! /usr/bin/env python
from src.rag_bot import DocumentChunk, RAGBot, RetrievedChunk

class DummyFallbackBot(RAGBot):
    def __init__(self, use_fallback: bool):
        self.use_llm = True
        self.model_client = object()
        self.api_mode = "chat"
        self.use_extractive_fallback_on_llm_unknown = use_fallback

    def _apply_terms_map(self, query):
        return query, []

    def retrieve(self, query):
        return [
            RetrievedChunk(
                score=0.1,
                chunk=DocumentChunk(
                    source_id="doc1",
                    chunk_id="chunk1",
                    text="Тестовый контент",
                    metadata={},
                ),
            )
        ]

    def _restore_original_terms(self, retrieved, replacements):
        return retrieved

    def _compose_extractive_answer(self, query, retrieved, replacements):
        return "Краткий обзор из ближайших фрагментов."

    def build_prompt(self, query, retrieved, replacements):
        return []

    def call_llm(self, prompt_messages):
        return "Я не знаю"

    def post_filter(self, text):
        return True, text

    def _text_contains_blocked_terms(self, text):
        return False


def test_llm_unknown_returns_default_answer():
    bot = DummyFallbackBot(use_fallback=False)

    result = bot.answer("Какой-то вопрос")

    assert result["answer"] == "Я не знаю."
    assert result["source"] == []
    assert "fallback" in result["explain"]


def test_llm_unknown_fallback_can_be_enabled():
    bot = DummyFallbackBot(use_fallback=True)

    result = bot.answer("Какой-то вопрос")

    assert "Краткий обзор" in result["answer"]
    assert result["source"] == ["doc1"]
    assert "FAISS" in result["explain"]