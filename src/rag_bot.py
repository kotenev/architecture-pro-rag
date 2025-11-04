import os, json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.config import *
from src.utils import load_txt_files
from typing import List

from yandexcloud import SDK

class RAGBot:
    def __init__(self, index_dir=INDEX_DIR, embed_model=EMBED_MODEL, top_k=TOP_K):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self._load_index()
        # init Yandex SDK
        if not YANDEX_API_KEY:
            raise RuntimeError("YANDEX_API_KEY is not set")
        self.sdk = SDK(iam_token=YANDEX_API_KEY)
        self.llm = self.sdk.client("ai.foundation.models.v1.ModelService")

    def _load_index(self):
        meta = self.index_dir / "metadata.json"
        idx = self.index_dir / "faiss.index"
        if not meta.exists() or not idx.exists():
            raise FileNotFoundError("FAISS index not found — run build_index.py first.")
        with open(meta, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.chunks = self.metadata["chunks"]
        self.index = faiss.read_index(str(idx))
        print(f"Index loaded ({len(self.chunks)} chunks).")

    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        return v.astype("float32")

    def retrieve(self, query: str):
        qv = self._embed_query(query)
        D, I = self.index.search(qv, self.top_k)
        res = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks): continue
            res.append({"score": float(dist), "chunk": self.chunks[idx]})
        return res

    def build_prompt(self, query: str, retrieved: List[dict], fewshot_examples: List[dict]=None):
        system = (
            "System: Ты технический помощник. "
            "Сначала рассуждай по шагам (Chain-of-Thought), затем выводи ответ. "
            "Если информации недостаточно, честно скажи: 'Я не знаю'.\n\n"
        )

        context = "Контекст из базы знаний:\n"
        for r in retrieved:
            c = r["chunk"]
            context += f"- Источник {c['source_id']}:\n{c['text']}\n\n"

        fewshot_text = ""
        if fewshot_examples:
            fewshot_text += "Примеры (Few-shot):\n"
            for ex in fewshot_examples:
                fewshot_text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

        prompt = f"{system}{fewshot_text}{context}Вопрос: {query}\nОтвет:"
        return prompt

    def call_llm(self, prompt: str, max_tokens=512, temperature=0.3):
        req = {
            "model_uri": f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_LLM_MODEL}/latest",
            "completion_options": {"stream": False, "temperature": temperature, "maxTokens": max_tokens},
            "messages": [{"role": "user", "text": prompt}],
        }
        resp = self.llm.completion(**req)
        return resp.result.alternatives[0].message.text.strip()

    def post_filter(self, text: str):
        lower = text.lower()
        for bad in SAFETY_BLOCKLIST:
            if bad in lower:
                return False, f"Filtered potentially unsafe phrase: {bad}"
        return True, text

    def answer(self, query: str, fewshot_examples: List[dict]=None):
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": "Я не знаю.", "source": [], "explain": "Нет релевантных фрагментов."}

        prompt = self.build_prompt(query, retrieved, fewshot_examples)
        try:
            llm_out = self.call_llm(prompt)
        except Exception as e:
            return {"answer": "Ошибка обращения к LLM", "source": [], "explain": str(e)}

        ok, filtered = self.post_filter(llm_out)
        if not ok:
            return {"answer": "Я не могу ответить (фильтрация безопасности).", "source": [r['chunk']['source_id'] for r in retrieved], "explain": filtered}

        return {"answer": filtered, "source": [r['chunk']['source_id'] for r in retrieved], "explain": "OK"}
