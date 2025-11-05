import json, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from src.config import *
from src.utils import load_txt_files
from typing import List

from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.auth import APIKeyAuth

sdk = YCloudML(
    folder_id=YANDEX_FOLDER_ID,
    auth=APIKeyAuth(YANDEX_API_KEY)
)

class RAGBot:
    def __init__(self, index_dir=INDEX_DIR, embed_model=EMBED_MODEL, top_k=TOP_K):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self._load_index()

    def _load_index(self):
        meta_path = self.index_dir / "metadata.json"
        idx_path = self.index_dir / "faiss.index"
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError("Index or metadata not found. Run build_index.py first.")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.chunks = self.metadata["chunks"]
        self.index = faiss.read_index(str(idx_path))
        print("Loaded FAISS index with", len(self.chunks), "chunks.")

    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        return v.astype("float32")

    def retrieve(self, query: str):
        qv = self._embed_query(query)
        D, I = self.index.search(qv, self.top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks): continue
            results.append({"score": float(dist), "chunk": self.chunks[idx]})
        return results

    def build_prompt(self, query: str, retrieved: List[dict], fewshot_examples: List[dict]=None):
        # System prompt with CoT instruction
        system = (
            "System: Ты ассистент техдоков компании. Сначала коротко опиши шаги рассуждения (Chain-of-Thought), "
            "потом дай окончательный ответ. Если в базе нет точного ответа — честно скажи: 'Я не знаю'.\n\n"
        )

        # Context assembly (top N chunks, with source citations)
        context = "Найденные фрагменты (с цитированием):\n"
        for r in retrieved:
            c = r["chunk"]
            context += f"- Источник: {c['source_id']}, id={c['chunk_id']}\n{c['text']}\n\n"

        # Few-shot (if provided)
        fewshot_text = ""
        if fewshot_examples:
            fewshot_text += "Примеры:\n"
            for ex in fewshot_examples:
                fewshot_text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

        # Query block
        prompt = system + fewshot_text + context + f"Вопрос: {query}\nОтвет (с шагами рассуждения):"
        return prompt

    def call_llm(self, prompt: str, max_tokens=512, temperature=0.0):
        # OpenAI chat completion example (adjust if using other provider)
        model = sdk.models.completions("yandexgpt")
        model = model.configure(temperature=temperature)
        resp = model.run(prompt)
        return resp #["choices"][0]["message"]["content"]

    def post_filter(self, text: str):
        # simple safety: block if any blocklist phrase occurs
        lower = text.lower()
        for bad in SAFETY_BLOCKLIST:
            if bad in lower:
                return False, f"Filtered potential secret or unsafe content: contains '{bad}'"
        return True, text

    def answer(self, query: str, fewshot_examples: List[dict]=None):
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": "Я не знаю.", "source": [], "explain": "Нет найденных фрагментов."}
        prompt = self.build_prompt(query, retrieved, fewshot_examples=fewshot_examples)
        llm_out = self.call_llm(prompt)
        ok, filtered = self.post_filter(llm_out)
        if not ok:
            return {"answer": "Я не могу предоставить информацию — потенциально небезопасно.", "source": [r["chunk"]["source_id"] for r in retrieved], "explain": filtered}
        return {"answer": filtered, "source": [r["chunk"]["source_id"] for r in retrieved], "explain": "OK"}

