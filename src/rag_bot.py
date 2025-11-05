#import os
import json
#from pathlib import Path
from typing import List

#import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline #, AutoTokenizer, AutoModelForCausalLM

from src.config import *
#from src.utils import load_txt_files


class RAGBot:
    """
    Локальный RAG-бот:
      • эмбеддинги: SentenceTransformers (MiniLM)
      • поиск: FAISS
      • LLM: локальная HuggingFace модель (pipeline)
    """

    def __init__(
        self,
        index_dir=INDEX_DIR,
        embed_model=EMBED_MODEL,
        top_k=TOP_K,
        llm_model_name=None,
        device_map="auto",
    ):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k

        self._load_index()

        self.llm_model_name = llm_model_name or os.environ.get(
            "LOCAL_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        print(f"Загружается локальная LLM: {self.llm_model_name}")
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model_name,
            tokenizer=self.llm_model_name,
            device_map=device_map,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=False,
        )

    def _load_index(self):
        meta_path = self.index_dir / "metadata.json"
        idx_path = self.index_dir / "faiss.index"
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError("Индекс не найден. Сначала запустите build_index.py")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.chunks = self.metadata["chunks"]
        self.index = faiss.read_index(str(idx_path))
        print(f"Индекс загружен: {len(self.chunks)} чанков")

    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        return v.astype("float32")

    def retrieve(self, query: str):
        qv = self._embed_query(query)
        D, I = self.index.search(qv, self.top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({"score": float(dist), "chunk": self.chunks[idx]})
        return results

    def build_prompt(self, query: str, retrieved: List[dict], fewshot_examples: List[dict] = None):
        """
        Формирует итоговый промпт:
         - System-инструкция с Chain-of-Thought
         - Few-shot примеры (если заданы)
         - Топ-чанки из базы
         - Вопрос пользователя
        """
        system_prompt = (
            "System: Ты технический ассистент, который сначала размышляет по шагам, "
            "а потом формулирует краткий ответ. "
            "Если данных недостаточно — скажи: 'Я не знаю'.\n\n"
        )

        context = "Контекст (фрагменты базы знаний):\n"
        for r in retrieved:
            c = r["chunk"]
            context += f"- [{c['source_id']}]: {c['text']}\n\n"

        fewshot_text = ""
        if fewshot_examples:
            fewshot_text += "Примеры (Few-shot):\n"
            for ex in fewshot_examples:
                fewshot_text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

        prompt = f"{system_prompt}{fewshot_text}{context}Вопрос: {query}\nОтвет (с шагами рассуждения):"
        return prompt

    def call_llm(self, prompt: str) -> str:
        """Генерация ответа локальной LLM через pipeline."""
        outputs = self.llm_pipeline(prompt)
        return outputs[0]["generated_text"].strip()

    def post_filter(self, text: str):
        lower = text.lower()
        for bad in SAFETY_BLOCKLIST:
            if bad in lower:
                return False, f"Отфильтровано: найдено '{bad}'"
        return True, text

    def answer(self, query: str, fewshot_examples: List[dict] = None):
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": "Я не знаю.", "source": [], "explain": "Нет релевантных фрагментов"}

        prompt = self.build_prompt(query, retrieved, fewshot_examples)
        try:
            raw = self.call_llm(prompt)
        except Exception as e:
            return {"answer": "Ошибка локальной модели", "source": [], "explain": str(e)}

        ok, filtered = self.post_filter(raw)
        if not ok:
            return {
                "answer": "Я не могу ответить (фильтрация безопасности).",
                "source": [r["chunk"]["source_id"] for r in retrieved],
                "explain": filtered,
            }

        return {
            "answer": filtered,
            "source": [r["chunk"]["source_id"] for r in retrieved],
            "explain": "OK",
        }
