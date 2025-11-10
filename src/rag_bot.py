#! /usr/bin/env python
import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from src.config import *

from yandex_cloud_ml_sdk import YCloudML

logger = logging.getLogger(__name__)


class RAGBot:

    def __init__(
            self,
            index_dir: str = INDEX_DIR,
            embed_model: str = EMBED_MODEL,
            top_k: int = TOP_K,
            use_llm: bool = True,
    ):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self._load_index()
        self.terms_map, self._terms_regex, self._terms_lookup = self._load_terms_map()
        self.api_mode: Optional[str] = None
        self.model_client = None
        self.sdk: Optional[YCloudML] = None
        self.yandex_model: Optional[str] = None

        self.use_llm = bool(use_llm)
        if self.use_llm and not (YANDEX_FOLDER_ID and YANDEX_API_KEY):
            logger.warning(
                "YANDEX_FOLDER_ID или YANDEX_API_KEY не заданы — работа без LLM"
            )
            self.use_llm = False
        if self.use_llm:
            self._init_llm_client()

    def _load_terms_map(self) -> Tuple[Dict[str, str], Optional[re.Pattern], Dict[str, str]]:
        terms_map: Dict[str, str] = {}
        terms_lookup: Dict[str, str] = {}

        if not TERMS_MAP_FILE:
            return terms_map, None, terms_lookup

        path = Path(TERMS_MAP_FILE)
        if not path.exists():
            return terms_map, None, terms_lookup

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Не удалось загрузить terms_map.json: {exc}") from exc

        if not isinstance(raw_map, dict):
            raise RuntimeError("Файл terms_map.json имеет некорректный формат")

        for original, internal in raw_map.items():
            if not isinstance(original, str) or not isinstance(internal, str):
                continue
            cleaned_original = original.strip()
            cleaned_internal = internal.strip()
            if not cleaned_original or not cleaned_internal:
                continue
            terms_map[cleaned_original] = cleaned_internal
            terms_lookup[cleaned_original.lower()] = cleaned_internal

        if not terms_map:
            return terms_map, None, terms_lookup

        pattern = "|".join(sorted((re.escape(k) for k in terms_map.keys()), key=len, reverse=True))
        regex = re.compile(rf"\b({pattern})\b", flags=re.IGNORECASE)
        return terms_map, regex, terms_lookup

    def _init_llm_client(self) -> None:
        if not (YANDEX_FOLDER_ID and YANDEX_API_KEY):
            raise RuntimeError(
                "Для использования LLM необходимо указать YANDEX_FOLDER_ID и YANDEX_API_KEY"
            )

        self.sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
        raw_model_name = YANDEX_LLM_MODEL or "yandexgpt-5-lite"
        self.yandex_model = self._resolve_model_uri(raw_model_name)

        models_obj = getattr(self.sdk, "models", None)
        if not models_obj:
            raise RuntimeError(
                "SDK не содержит атрибут 'models' — обновите yandex-cloud-ml-sdk"
            )

        if hasattr(models_obj, "chat"):
            logger.info("Используется режим CHAT (sdk.models.chat)")
            model_builder = models_obj.chat
            self.api_mode = "chat"
        elif hasattr(models_obj, "completions"):
            logger.info("Используется режим COMPLETIONS (sdk.models.completions)")
            model_builder = models_obj.completions
            self.api_mode = "completions"
        else:
            raise RuntimeError("SDK не имеет методов chat или completions")

        self.model_client = model_builder(self.yandex_model).configure(
            temperature=0.0,
            max_tokens=1024,
        )

    def _resolve_model_uri(self, model_value: str) -> str:

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

    def _embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(v)
        return v.astype("float32")

    def _apply_terms_map(self, query: str) -> Tuple[str, Sequence[Tuple[str, str]]]:
        if not self._terms_regex:
            return query, []

        replacements: List[Tuple[str, str]] = []

        def _replacer(match: re.Match) -> str:
            found = match.group(0)
            replacement = self._terms_lookup.get(found.lower())
            if replacement:
                replacements.append((found, replacement))
                return replacement
            return found

        normalized_query = self._terms_regex.sub(_replacer, query)
        return normalized_query, replacements

    def retrieve(self, query: str):
        qv = self._embed_query(query)
        D, I = self.index.search(qv, self.top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.chunks):
                results.append({"score": float(dist), "chunk": self.chunks[idx]})
        return results

    def _text_contains_blocked_terms(self, text: str) -> bool:
        lower = text.lower()
        return any(bad.lower() in lower for bad in SAFETY_BLOCKLIST)

    def _iter_sentences(self, text: str) -> Iterable[str]:
        pieces = re.split(r"(?<=[.!?])\s+", text.strip())
        for piece in pieces:
            sentence = piece.strip()
            if sentence:
                yield sentence

    def _compose_extractive_answer(
        self,
        query: str,
        retrieved: Sequence[dict],
        term_mappings: Sequence[Tuple[str, str]],
    ) -> Optional[str]:
        focus_terms: List[str] = []
        for original, mapped in term_mappings:
            if original:
                focus_terms.append(original)
            if mapped:
                focus_terms.append(mapped)
        if not focus_terms:
            focus_terms = [query]

        focus_terms_lower = [term.lower() for term in focus_terms]

        collected: List[str] = []
        for item in retrieved:
            chunk = item.get("chunk", {})
            text = chunk.get("text", "").strip()
            if not text or self._text_contains_blocked_terms(text):
                continue

            sentences = list(self._iter_sentences(text))
            if not sentences:
                continue

            matching = [
                sent for sent in sentences
                if any(term in sent.lower() for term in focus_terms_lower)
            ]

            selected = matching or sentences[:2]

            for sent in selected:
                if sent not in collected:
                    collected.append(sent)
            if len(collected) >= 4:
                break

        if not collected:
            return None

        answer = " ".join(collected[:4])
        if term_mappings:
            mapping_note = "; ".join(
                f"{orig} → {mapped}" for orig, mapped in term_mappings if orig and mapped
            )
            if mapping_note:
                answer += f"\n\nСопоставление терминов: {mapping_note}"
        return answer

    def build_prompt(
            self,
            query: str,
            retrieved: List[dict],
            fewshot_examples: List[dict] = None,
            term_mappings: Sequence[Tuple[str, str]] = (),
    ):
        system_prompt = (
            "Ты — технический помощник. "
            "Сначала рассуждай пошагово (Chain-of-Thought), затем дай краткий ответ. "
            "Если информации недостаточно, скажи 'Я не знаю'.\n\n"
        )

        mapping_note = ""
        if term_mappings:
            mapping_note_lines = [
                "Сопоставление терминов пользователя с внутренними идентификаторами:",
            ]
            mapping_note_lines.extend(
                f"- {orig} → {mapped}" for orig, mapped in term_mappings
            )
            mapping_note_lines.append(
                "Если в контексте встречается внутренний идентификатор, считай, что речь идёт об исходном термине пользователя."
            )
            mapping_note = "\n".join(mapping_note_lines) + "\n\n"

        context = "Контекст из базы знаний:\n"
        for r in retrieved:
            c = r["chunk"]
            context += f"- [{c['source_id']}]: {c['text']}\n\n"

        fewshot_text = ""
        if fewshot_examples:
            fewshot_text += "Примеры (Few-shot):\n"
            for ex in fewshot_examples:
                fewshot_text += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

        return f"{system_prompt}{mapping_note}{fewshot_text}{context}Вопрос: {query}\nОтвет:".strip()

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
                return False, f"Фильтр безопасности: найдено '{bad}'"
        return True, text

    def _restore_original_terms(
        self,
        retrieved: List[dict],
        term_mappings: Sequence[Tuple[str, str]],
    ) -> List[dict]:

        if not term_mappings:
            return retrieved

        reverse_map = {}
        for original, internal in term_mappings:
            internal_clean = internal.strip()
            original_clean = original.strip()
            if not internal_clean or not original_clean:
                continue
            reverse_map[internal_clean.lower()] = original_clean

        if not reverse_map:
            return retrieved

        restored: List[dict] = []
        for item in retrieved:
            chunk = dict(item["chunk"])
            text = chunk.get("text", "")
            for internal_lower, original in reverse_map.items():
                pattern = re.compile(rf"\b{re.escape(internal_lower)}\b", flags=re.IGNORECASE)
                text = pattern.sub(original, text)
            chunk["text"] = text
            restored.append({"score": item["score"], "chunk": chunk})

        return restored

    def answer(self, query: str, fewshot_examples: List[dict] = None):
        normalized_query, replacements = self._apply_terms_map(query)
        retrieved = self.retrieve(normalized_query)


        mapped_terms = [
            {"original": orig, "internal": mapped}
            for orig, mapped in replacements
            if orig and mapped
        ]

        def build_response(answer_text, sources, explain_text):
            return {
                "answer": answer_text,
                "source": sources,
                "explain": explain_text,
                "terms": mapped_terms,
            }

        if not retrieved:
            return build_response(
                "Я не знаю.",
                [],
                "Нет релевантных фрагментов.",
            )

        retrieved_for_prompt = self._restore_original_terms(retrieved, replacements)
        source_ids = list(dict.fromkeys(r["chunk"]["source_id"] for r in retrieved))

        extractive_answer = self._compose_extractive_answer(
            query, retrieved_for_prompt, replacements
        )

        if self.use_llm and self.model_client is not None:
            prompt = self.build_prompt(
                query, retrieved_for_prompt, fewshot_examples, replacements
            )
            try:
                llm_out = self.call_llm(prompt)
            except Exception as e:
                return build_response(
                    "Ошибка при обращении к LLM",
                    source_ids,
                    str(e),
                )

            ok, filtered = self.post_filter(llm_out)
            if not ok:
                return build_response(
                    "Я не могу ответить (фильтрация безопасности).",
                    source_ids,
                    filtered,
                )

            normalized_answer = filtered.lower()
            unknown_markers = [
                "я не знаю",
                "нет информации",
                "информации нет",
                "не найдено",
                "i don't know",
            ]
            if any(marker in normalized_answer for marker in unknown_markers):
                if extractive_answer:
                    return build_response(
                        extractive_answer,
                        source_ids,
                        "Ответ построен на основе ближайших фрагментов FAISS.",
                    )
                return build_response(
                    "Я не знаю.",
                    [],
                    "LLM не нашла ответа и не удалось построить краткий обзор.",
                )

            if replacements and "сопоставление терминов" not in normalized_answer:
                mapping_note = "; ".join(
                    f"{orig} → {mapped}" for orig, mapped in replacements if orig and mapped
                )
                if mapping_note:
                    filtered = f"{filtered}\n\nСопоставление терминов: {mapping_note}"

            return build_response(filtered, source_ids, "OK")

        if not extractive_answer:
            return build_response(
                "Я не знаю.",
                [],
                "Не удалось подобрать релевантные предложения.",
            )

        if self._text_contains_blocked_terms(extractive_answer):
            return build_response(
                "Я не могу ответить (фильтрация безопасности).",
                [],
                "Ответ отклонён правилами безопасности.",
            )

        return build_response(extractive_answer, source_ids, "OK")
