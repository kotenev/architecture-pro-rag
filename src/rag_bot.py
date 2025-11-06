import json
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from src.config import *

from yandex_cloud_ml_sdk import YCloudML


class RAGBot:

    def __init__(self, index_dir: str = INDEX_DIR, embed_model: str = EMBED_MODEL, top_k: int = TOP_K):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self._load_index()

        if not (YANDEX_FOLDER_ID and YANDEX_API_KEY):
            raise RuntimeError("YANDEX_FOLDER_ID или YANDEX_API_KEY не заданы (см. .env или config.py)")

        self.sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
        raw_model_name = YANDEX_LLM_MODEL or "yandexgpt-5-lite"
        self.yandex_model = self._resolve_model_uri(raw_model_name)

        models_obj = getattr(self.sdk, "models", None)
        if not models_obj:
            raise RuntimeError("SDK не содержит атрибут 'models' — обновите yandex-cloud-ml-sdk")

        if hasattr(models_obj, "chat"):
            print("Используется режим CHAT (sdk.models.chat)")
            model_builder = models_obj.chat
            self.api_mode = "chat"
        elif hasattr(models_obj, "completions"):
            print("Используется режим COMPLETIONS (sdk.models.completions)")
            model_builder = models_obj.completions
            self.api_mode = "completions"
        else:
            raise RuntimeError("SDK не имеет методов chat или completions")

        self.model_client = model_builder(self.yandex_model).configure(
            temperature=0.0,
            max_tokens=1024,
        )


    def _resolve_model_uri(self, model_value: str) -> str:
        """Convert short model names to full YandexGPT URIs."""

        if not model_value:
            raise RuntimeError("YANDEX_LLM_MODEL не задан")

        model_value = model_value.strip()
        if model_value.startswith("gpt://"):
            return model_value

        if not YANDEX_FOLDER_ID:
            raise RuntimeError(
                "Для короткой формы YANDEX_LLM_MODEL необходимо задать YANDEX_FOLDER_ID"
            )

        for delimiter in ("@", ":"):
            if delimiter in model_value:
                model_name, version = model_value.split(delimiter, 1)
                break
        else:
            model_name, version = model_value, "latest"

        model_name = model_name.strip("/ ")
        version = version.strip() or "latest"

        if not model_name:
            raise RuntimeError("Некорректное значение YANDEX_LLM_MODEL")

        return f"gpt://{YANDEX_FOLDER_ID}/{model_name}/{version}"


    def _load_index(self):
        meta_path = Path(self.index_dir) / "metadata.json"
        idx_path = Path(self.index_dir) / "faiss.index"
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError("Индекс не найден. Запустите build_index.py")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.chunks = self.metadata["chunks"]
        self.index = faiss.read_index(str(idx_path))
        print(f"FAISS индекс загружен: {len(self.chunks)} чанков")

    # ---------- Эмбеддинг и поиск ----------
    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        return v.astype("float32")

    def retrieve(self, query: str):
        qv = self._embed_query(query)
        D, I = self.index.search(qv, self.top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.chunks):
                results.append({"score": float(dist), "chunk": self.chunks[idx]})
        return results

    # ---------- Промпт ----------
    def build_prompt(self, query: str, retrieved: List[dict], fewshot_examples: List[dict] = None):
        system_prompt = (
            "Ты — технический помощник. "
            "Сначала рассуждай пошагово (Chain-of-Thought), затем дай краткий ответ. "
            "Если информации недостаточно, скажи 'Я не знаю'.\n\n"
        )

        context = "Контекст из базы знаний:\n"
        for r in retrieved:
            c = r["chunk"]
            context += f"- [{c['source_id']}]: {c['text']}\n\n"

        fewshot_text = ""
        if fewshot_examples:
            fewshot_text += "Примеры (Few-shot):\n"
            for ex in fewshot_examples:
                fewshot_text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

        return f"{system_prompt}{fewshot_text}{context}Вопрос: {query}\nОтвет:".strip()

    def call_llm(self, prompt_or_messages):
        try:
            if self.api_mode == "chat":
                if isinstance(prompt_or_messages, str):
                    messages = [{"role": "user", "text": prompt_or_messages}]
                else:
                    messages = prompt_or_messages
                result = self.model_client.run(messages)
            else:
                if not isinstance(prompt_or_messages, str):
                    prompt = "\n".join(m.get("text", "") for m in prompt_or_messages)
                else:
                    prompt = prompt_or_messages
                result = self.model_client.run(prompt)
        except Exception as e:
            raise RuntimeError(f"Ошибка вызова модели: {e}")

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
        return text.strip()

    def post_filter(self, text: str):
        lower = text.lower()
        for bad in SAFETY_BLOCKLIST:
            if bad in lower:
                return False, f"⚠️ Фильтр безопасности: найдено '{bad}'"
        return True, text

    def answer(self, query: str, fewshot_examples: List[dict] = None):
        retrieved = self.retrieve(query)
        if not retrieved:
            return {"answer": "Я не знаю.", "source": [], "explain": "Нет релевантных фрагментов."}

        prompt = self.build_prompt(query, retrieved, fewshot_examples)
        try:
            llm_out = self.call_llm(prompt)
        except Exception as e:
            return {
                "answer": "Ошибка при обращении к LLM",
                "source": [r["chunk"]["source_id"] for r in retrieved],
                "explain": str(e),
            }

        ok, filtered = self.post_filter(llm_out)
        if not ok:
            return {
                "answer": "Я не могу ответить (фильтрация безопасности).",
                "source": [r["chunk"]["source_id"] for r in retrieved],
                "explain": filtered,
            }

        return {"answer": filtered, "source": [r["chunk"]["source_id"] for r in retrieved], "explain": "OK"}
