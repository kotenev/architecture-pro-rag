#! /usr/bin/env python
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from yandex_cloud_ml_sdk import YCloudML

from .config import (
    EMBED_DIM,
    EMBED_MODEL,
    FEWSHOT_FILE,
    INDEX_DIR,
    SAFETY_BLOCKLIST,
    TERMS_MAP_FILE,
    TOP_K,
    YANDEX_API_KEY,
    YANDEX_FOLDER_ID,
    YANDEX_LLM_MODEL,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DocumentChunk:
    source_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True)
class FewShotExample:
    question: str
    answer: str

class RAGBot:

    def __init__(
            self,
            index_dir: str = INDEX_DIR,
            embed_model: str = EMBED_MODEL,
            top_k: int = TOP_K,
            fewshot_file: str = FEWSHOT_FILE,
            use_llm: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model
        self.top_k = max(1, int(top_k))
        self.fewshot_file = fewshot_file

        self.model = SentenceTransformer(self.embed_model_name)
        self.index, self.chunks = self._load_index(self.index_dir)
        self.terms_map, self._terms_regex, self._terms_lookup = self._load_terms_map(
            TERMS_MAP_FILE
        )
        self.fewshot_examples = self._load_fewshot_examples(self.fewshot_file)

        self.use_llm = bool(use_llm)
        self.api_mode: Optional[str] = None
        self.model_client = None
        self.sdk: Optional[YCloudML] = None
        self.yandex_model: Optional[str] = None

        if self.use_llm and not (YANDEX_FOLDER_ID and YANDEX_API_KEY):
            logger.warning(
                "YANDEX_FOLDER_ID или YANDEX_API_KEY не заданы — работа без LLM"
            )
            self.use_llm = False

        if self.use_llm:
            self._init_llm_client()

    # ------------------------------------------------------------------
    # Инициализация и загрузка ресурсов
    # ------------------------------------------------------------------
    def _load_index(
        self, index_dir: Path
    ) -> Tuple[faiss.Index, List[DocumentChunk]]:
        meta_path = index_dir / "metadata.json"
        idx_path = index_dir / "faiss.index"

        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                "Индекс не найден. Запустите build_index.py для создания индекса"
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        raw_chunks = metadata.get("chunks", [])
        chunks: List[DocumentChunk] = []
        for raw in raw_chunks:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", ""))
            chunk = DocumentChunk(
                source_id=str(raw.get("source_id", "unknown")),
                chunk_id=str(raw.get("chunk_id", "")),
                text=text,
                metadata={
                    key: value
                    for key, value in raw.items()
                    if key not in {"text", "source_id", "chunk_id"}
                },
            )
            chunks.append(chunk)

        index = faiss.read_index(str(idx_path))
        if index.d != EMBED_DIM:
            logger.warning(
                "Размерность индекса (%s) не совпадает с ожидаемой (%s)",
                index.d,
                EMBED_DIM,
            )

        logger.info("FAISS индекс загружен: %s чанков", len(chunks))
        return index, chunks

    def _load_terms_map(
        self, terms_file: Optional[str]
    ) -> Tuple[Dict[str, str], Optional[re.Pattern], Dict[str, str]]:
        terms_map: Dict[str, str] = {}
        terms_lookup: Dict[str, str] = {}

        if not terms_file:
            return terms_map, None, terms_lookup

        path = Path(terms_file)
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

        pattern = "|".join(
            sorted((re.escape(k) for k in terms_map.keys()), key=len, reverse=True)
        )
        regex = re.compile(rf"\b({pattern})\b", flags=re.IGNORECASE)
        return terms_map, regex, terms_lookup

    def _load_fewshot_examples(self, fewshot_file: Optional[str]) -> List[FewShotExample]:
        if not fewshot_file:
            return []

        path = Path(fewshot_file)
        if not path.exists():
            logger.warning("Файл с few-shot примерами не найден: %s", fewshot_file)
            return []

        examples: List[FewShotExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning("Некорректная строка few-shot: %s", exc)
                    continue

                question = str(payload.get("q", "")).strip()
                answer = payload.get("a", "")
                if isinstance(answer, list):
                    answer_text = "\n".join(str(item) for item in answer)
                else:
                    answer_text = str(answer)
                answer_text = answer_text.strip()

                if question and answer_text:
                    examples.append(FewShotExample(question=question, answer=answer_text))

        if not examples:
            logger.warning(
                "Не удалось загрузить ни одного few-shot примера из %s",
                fewshot_file,
            )

        return examples

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

    def _embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], convert_to_numpy=True)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector.astype("float32")

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

    def _search_index(
        self, query_vector: np.ndarray, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        limit = top_k or self.top_k
        distances, indices = self.index.search(query_vector, limit)
        results: List[RetrievedChunk] = []
        if len(distances) == 0 or len(indices) == 0:
            return results

        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append(
                    RetrievedChunk(
                        score=float(dist),
                        chunk=self.chunks[idx],
                    )
                )
        return results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        query_vector = self._embed_query(query)
        return self._search_index(query_vector, top_k)

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
        retrieved: Sequence[RetrievedChunk],
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
            text = item.chunk.text.strip()
            if not text or self._text_contains_blocked_terms(text):
                continue

            sentences = list(self._iter_sentences(text))
            if not sentences:
                continue

            matching = [
                sent
                for sent in sentences
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

    def _restore_original_terms(
        self,
        retrieved: Sequence[RetrievedChunk],
        term_mappings: Sequence[Tuple[str, str]],
    ) -> List[RetrievedChunk]:
        if not term_mappings:
            return list(retrieved)

        reverse_map: Dict[str, str] = {}
        for original, internal in term_mappings:
            internal_clean = internal.strip()
            original_clean = original.strip()
            if not internal_clean or not original_clean:
                continue
            reverse_map[internal_clean.lower()] = original_clean

        if not reverse_map:
            return list(retrieved)

        restored: List[RetrievedChunk] = []
        for item in retrieved:
            text = item.chunk.text
            for internal_lower, original in reverse_map.items():
                pattern = re.compile(rf"\b{re.escape(internal_lower)}\b", flags=re.IGNORECASE)
                text = pattern.sub(original, text)

            restored_chunk = DocumentChunk(
                source_id=item.chunk.source_id,
                chunk_id=item.chunk.chunk_id,
                text=text,
                metadata=item.chunk.metadata,
            )
            restored.append(RetrievedChunk(score=item.score, chunk=restored_chunk))

        return restored

    def _build_mapping_note(
        self, term_mappings: Sequence[Tuple[str, str]]
    ) -> str:
        if not term_mappings:
            return ""

        mapping_note_lines = [
            "Сопоставление терминов пользователя с внутренними идентификаторами:",
        ]
        mapping_note_lines.extend(
            f"- {orig} → {mapped}"
            for orig, mapped in term_mappings
            if orig and mapped
        )
        mapping_note_lines.append(
            "Если встречается внутренний идентификатор, считай, что речь об исходном термине."
        )
        return "\n".join(mapping_note_lines)

    def build_prompt(
            self,
            query: str,
            retrieved: Sequence[RetrievedChunk],
            term_mappings: Sequence[Tuple[str, str]] = (),
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги. "
            "Опирайся только на предоставленный контекст. Если данных недостаточно, отвечай: 'Я не знаю'."
        )

        mapping_note = self._build_mapping_note(term_mappings)

        context_lines: List[str] = []
        for idx, item in enumerate(retrieved, start=1):
            chunk = item.chunk
            context_lines.append(
                f"[{idx}. Источник: {chunk.source_id}] {chunk.text}"
            )
            context_block = "\n\n".join(context_lines) if context_lines else "Контекст не найден."

            fewshot_parts: List[str] = []
            for example in self.fewshot_examples[:2]:
                fewshot_parts.append(
                    f"Пример.\nQ: {example.question}\nA:\n{example.answer}"
                )
            fewshot_text = "\n\n".join(fewshot_parts)

        instructions = (
            "Сначала перечисли рассуждения в виде нумерованных шагов. "
            "После этого на отдельной строке напиши 'Ответ: ...'."
        )

        user_parts: List[str] = []
        if mapping_note:
            user_parts.append(mapping_note)
        if fewshot_text:
            user_parts.append(fewshot_text)
        user_parts.append(f"Контекст:\n{context_block}")
        user_parts.append(instructions)
        user_parts.append(f"Вопрос: {query}\n\nОтвет:")
        user_message = "\n\n".join(part for part in user_parts if part)

        return [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_message},
        ]

    def call_llm(self, prompt_or_messages: Sequence[Dict[str, str]]) -> str:
        if not self.model_client:
                raise RuntimeError("LLM клиент не инициализирован")

        try:
            if self.api_mode == "chat":
                messages = [
                    {"role": msg.get("role", "user"), "text": msg.get("text", "")}
                    for msg in prompt_or_messages
                ]
                result = self.model_client.run(messages)
            else:
                prompt = "\n\n".join(msg.get("text", "") for msg in prompt_or_messages)
                result = self.model_client.run(prompt)
        except Exception as exc:
            raise RuntimeError(f"Ошибка вызова модели: {exc}") from exc

        text = None
        try:
            if hasattr(result, "result") and hasattr(result.result, "alternatives"):
                text = result.result.alternatives[0].text
            elif isinstance(result, list) and result and hasattr(result[0], "text"):
                text = result[0].text
            elif hasattr(result, "alternatives"):
                text = result.alternatives[0].text
            else:
                text = str(result)
        except Exception:
            text = str(result)

        return (text or "").strip()

    def post_filter(self, text: str) -> Tuple[bool, str]:
        lower = text.lower()
        for bad in SAFETY_BLOCKLIST:
            if bad.lower() in lower:
                return False, f"Фильтр безопасности: найдено '{bad}'"
        return True, text

    def answer(self, query: str) -> Dict[str, Any]:
        normalized_query, replacements = self._apply_terms_map(query)
        retrieved = self.retrieve(normalized_query)

        if retrieved:
            logger.info("Найденные чанки (%d):", len(retrieved))
            for idx, item in enumerate(retrieved):
                chunk = item.chunk
                metadata_log = {"source": chunk.source_id, "chunk_id": chunk.chunk_id}
                metadata_log.update(chunk.metadata)
                logger.info(
                    "[%d] score = %.4f\nmetadata = %s\npage_content =\n\"%s\"",
                    idx,
                    item.score,
                    json.dumps(metadata_log, ensure_ascii=False, indent=2),
                    chunk.text.strip(),
                )

        mapped_terms = [
            {"original": orig, "internal": mapped}
            for orig, mapped in replacements
            if orig and mapped
        ]

        def build_response(answer_text: str, sources: List[str], explain_text: str):
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
               "Нет релевантных фрагментов в индексе.",
            )

        retrieved_for_prompt = self._restore_original_terms(retrieved, replacements)
        source_ids = list(dict.fromkeys(item.chunk.source_id for item in retrieved))

        extractive_answer = self._compose_extractive_answer(
            query, retrieved_for_prompt, replacements
        )

        if self.use_llm and self.model_client is not None:
            prompt_messages = self.build_prompt(query, retrieved_for_prompt, replacements)
            try:
                llm_out = self.call_llm(prompt_messages)
            except Exception as exc:
                return build_response(
                    "Ошибка при обращении к LLM",
                    source_ids,
                    str(exc),
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
