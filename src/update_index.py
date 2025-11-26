#!/usr/bin/env python

import json
import logging
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faker import Faker

from parse_fandom_pages import TermReplacement, replace_terms

from config import (
    EMBED_MODEL,
    INDEX_DIR,
    KB_DIR,
    FANDOM_PAGES_FILE,
    TERMS_MAP_FILE,
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IndexUpdater:
    def __init__(self):
        self.kb_dir = Path(KB_DIR)
        self.incoming_dir = self.kb_dir / "incoming"
        self.index_dir = Path(INDEX_DIR)
        self.processed_file = self.index_dir / "processed_files.json"
        self.fandom_pages_file = Path(FANDOM_PAGES_FILE) if FANDOM_PAGES_FILE else None

        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(EMBED_MODEL)

        self.fake = Faker("ru_RU")
        self.used_phrases: set = set()

        self.processed_files = self._load_processed_files()

        self.index, self.chunks, self.metadata = self._load_existing_index()

        self.terms_map = self._load_terms_map()
        self.used_phrases.update(self.terms_map.values())
        self.terms_map_updated = False
        self.term_replacements: List[TermReplacement] = self._build_term_replacements()
        self.fandom_pages = self._load_fandom_pages()
        self.fandom_pages_updated = False

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        self.stats = {
            "start_time": None,
            "end_time": None,
            "new_files": 0,
            "modified_files": 0,
            "new_chunks": 0,
            "deleted_chunks": 0,
            "total_chunks": 0,
            "errors": []
        }

    def _load_processed_files(self) -> Dict[str, str]:
        if self.processed_file.exists():
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        with open(self.processed_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=2)

    def _load_existing_index(self) -> Tuple:
        index_file = self.index_dir / "faiss.index"
        metadata_file = self.index_dir / "metadata.json"

        if index_file.exists() and metadata_file.exists():
            logger.info("Загрузка существующего индекса...")
            index = faiss.read_index(str(index_file))

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            chunks = metadata.get("chunks", [])
            logger.info(f"Загружено {len(chunks)} существующих чанков")
            return index, chunks, metadata
        else:
            logger.info("Создание нового индекса...")
            dim = 1024
            index = faiss.IndexFlatIP(dim)
            return index, [], {"chunks": [], "embed_dim": dim, "model": EMBED_MODEL}

    def _load_terms_map(self) -> Dict[str, str]:
        if TERMS_MAP_FILE and Path(TERMS_MAP_FILE).exists():
            with open(TERMS_MAP_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_fandom_pages(self) -> Dict[str, str]:
        if self.fandom_pages_file and self.fandom_pages_file.exists():
            with open(self.fandom_pages_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _build_term_replacements(self) -> List[TermReplacement]:
        replacements: List[TermReplacement] = []
        for original, replacement in self.terms_map.items():
            try:
                replacements.append(TermReplacement(original, replacement))
            except ValueError as exc:
                logger.warning(
                    "Не удалось добавить замену для '%s': %s", original, exc
                )
        return replacements

    def _save_terms_map(self):
        if not TERMS_MAP_FILE:
            return

        with open(TERMS_MAP_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.terms_map, f, ensure_ascii=False, indent=4)

    def _save_fandom_pages(self):
        if not self.fandom_pages_file:
            return

        self.fandom_pages_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fandom_pages_file, 'w', encoding='utf-8') as f:
            json.dump(self.fandom_pages, f, ensure_ascii=False, indent=4)

    def _generate_replacement_phrase(self, term: str) -> str:
        word_count = max(1, len(re.findall(r'[А-Яа-яЁё-]+', term)))

        def unique_first_name() -> str:
            return self.fake.unique.first_name()

        phrase = None
        attempts = 0
        while phrase is None or phrase in self.used_phrases:
            attempts += 1
            if attempts > 100:
                self.fake.unique.clear()
                attempts = 0

            if word_count == 1:
                phrase = unique_first_name()
            elif word_count == 2:
                phrase = f"{unique_first_name()} {self.fake.unique.last_name()}"
            else:
                words = [unique_first_name() for _ in range(word_count)]
                phrase = ' '.join(words)

        self.used_phrases.add(phrase)
        return phrase

    def _ensure_term_mapping(self, original_term: str) -> str:
        if original_term in self.terms_map:
            return self.terms_map[original_term]

        replacement = self._generate_replacement_phrase(original_term)
        self.terms_map[original_term] = replacement
        self.terms_map_updated = True
        try:
            self.term_replacements.append(TermReplacement(original_term, replacement))
        except ValueError as exc:
            logger.warning(
                "Не удалось создать замену для '%s': %s", original_term, exc
            )
        return replacement

    def _register_fandom_page(self, filepath: Path):
        if not self.fandom_pages_file:
            return

        url_file = self.incoming_dir / f"{filepath.stem}.url"
        if not url_file.exists():
            return

        url = url_file.read_text(encoding='utf-8').strip()
        if not url:
            return

        existing_url = self.fandom_pages.get(filepath.stem)
        if existing_url:
            if existing_url != url:
                logger.info(
                    "URL для '%s' уже зафиксирован и отличается от входного: %s",
                    filepath.stem,
                    existing_url,
                )
            return

        self.fandom_pages[filepath.stem] = url
        self.fandom_pages_updated = True
        logger.info(
            "Добавлено новое соответствие термина '%s' и URL %s",
            filepath.stem,
            url,
        )

    def _calculate_file_hash(self, filepath: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _apply_terms_replacement(self, text: str) -> str:
        if not self.term_replacements:
            return text

        return replace_terms(text, self.term_replacements)

    def _transform_file_content(self, filepath: Path) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._apply_terms_replacement(text)

    def _find_new_and_modified_files(self) -> Tuple[List[Path], List[Path], List[Path]]:
        new_files = []
        modified_files = []
        unchanged_files = []

        for filepath in self.incoming_dir.glob("*.txt"):
            file_hash = self._calculate_file_hash(filepath)
            filename = filepath.name

            if filename not in self.processed_files:
                new_files.append(filepath)
                logger.info(f"Новый файл: {filename}")
            elif self.processed_files[filename] != file_hash:
                modified_files.append(filepath)
                logger.info(f"Изменённый файл: {filename}")
            else:
                unchanged_files.append(filepath)
                logger.info(f"Файл без изменений: {filename}")

        return new_files, modified_files, unchanged_files

    def _process_document(self, filepath: Path, doc_id: str, filename: str) -> Tuple[List[Dict], str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            text = self._transform_file_content(filepath)

            chunks_text = self.text_splitter.split_text(text)

            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                chunks.append({
                    "source_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",
                    "text": chunk_text,
                    "filename": filename,
                    "processed_at": datetime.now().isoformat()
                })

            return chunks, text

        except Exception as e:
            logger.error(f"Ошибка обработки файла {filepath}: {e}")
            self.stats["errors"].append(f"{filepath.name}: {str(e)}")
            return [], ""

    def _remove_old_chunks(self, doc_id: str):
        indices_to_remove = []
        for i, chunk in enumerate(self.chunks):
            if chunk.get("source_id") == doc_id:
                indices_to_remove.append(i)

        if indices_to_remove:
            for i in reversed(indices_to_remove):
                del self.chunks[i]

            self.stats["deleted_chunks"] += len(indices_to_remove)
            logger.info(f"Удалено {len(indices_to_remove)} старых чанков для {doc_id}")

            self._rebuild_index()

    def _rebuild_index(self):
        if not self.chunks:
            return

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self._generate_embeddings(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype("float32")

    def _add_new_chunks(self, new_chunks: List[Dict]):
        if not new_chunks:
            return

        texts = [chunk["text"] for chunk in new_chunks]
        embeddings = self._generate_embeddings(texts)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.chunks.extend(new_chunks)
        self.stats["new_chunks"] += len(new_chunks)

        logger.info(f"Добавлено {len(new_chunks)} новых чанков")

    def _remove_related_url_file(self, filepath: Path):
        url_file = self.incoming_dir / f"{filepath.stem}.url"
        if url_file.exists():
            url_file.unlink()
            logger.info(
                f"Удалён связанный URL-файл {url_file.name} для {filepath.name}"
            )

    def _move_processed_file(self, filepath: Path, transformed_filename: str, transformed_text: str):
        destination = self.kb_dir / transformed_filename
        destination.write_text(transformed_text, encoding='utf-8')
        filepath.unlink()
        logger.info(
            f"Файл {filepath.name} преобразован и сохранён в {destination}"
        )

        self._remove_related_url_file(filepath)

    def _cleanup_unchanged_files(self, files: List[Path]):
        if not files:
            return

        for filepath in files:
            self._register_fandom_page(filepath)
            transformed_term = self._ensure_term_mapping(filepath.stem)
            logger.info(
                "Файл %s уже обработан ранее. Термин '%s' сохранён без изменений",
                filepath.name,
                transformed_term,
            )

            filepath.unlink()
            self._remove_related_url_file(filepath)

    def _update_kb_files_from_terms_map(self):
        for filepath in self.kb_dir.glob("*.txt"):
            if filepath.parent == self.incoming_dir:
                continue

            stem_variants = [filepath.stem, filepath.stem.replace('_', ' ')]
            original_term = None
            for variant in stem_variants:
                if variant in self.terms_map:
                    original_term = variant
                    break

            if not original_term:
                continue

            replacement = self.terms_map[original_term]
            transformed_filename = f"{replacement.replace(' ', '_')}{filepath.suffix}"
            destination = self.kb_dir / transformed_filename
            original_text = filepath.read_text(encoding='utf-8')
            transformed_text = self._apply_terms_replacement(original_text)

            needs_rename = filepath.name != transformed_filename
            needs_content_update = original_text != transformed_text

            if not needs_rename and not needs_content_update:
                continue

            destination.write_text(transformed_text, encoding='utf-8')

            if destination != filepath:
                filepath.unlink()

            logger.info(
                "Файл %s обновлён согласно текущему terms_map как %s",
                filepath.name,
                destination.name,
            )

    def _save_index(self):
        index_file = self.index_dir / "faiss.index"
        metadata_file = self.index_dir / "metadata.json"

        faiss.write_index(self.index, str(index_file))

        self.metadata["chunks"] = self.chunks
        self.metadata["last_update"] = datetime.now().isoformat()
        self.metadata["total_chunks"] = len(self.chunks)

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Индекс сохранён: {len(self.chunks)} чанков")

    def _save_update_log(self):
        log_file = LOG_DIR / f"update_summary_{datetime.now().strftime('%Y%m%d')}.json"

        stats_to_save = self.stats.copy()

        for key in ("start_time", "end_time"):
            if isinstance(stats_to_save.get(key), datetime):
                stats_to_save[key] = stats_to_save[key].isoformat()

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=2)

        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        logger.info("=" * 60)
        logger.info("ИТОГИ ОБНОВЛЕНИЯ:")
        logger.info(f"  Время выполнения: {duration:.2f} сек")
        logger.info(f"  Новых файлов: {self.stats['new_files']}")
        logger.info(f"  Изменённых файлов: {self.stats['modified_files']}")
        logger.info(f"  Новых чанков: {self.stats['new_chunks']}")
        logger.info(f"  Удалённых чанков: {self.stats['deleted_chunks']}")
        logger.info(f"  Всего чанков в индексе: {self.stats['total_chunks']}")
        logger.info(f"  Ошибок: {len(self.stats['errors'])}")
        logger.info("=" * 60)

    def run(self):
        self.stats["start_time"] = datetime.now()
        logger.info("Начало обновления индекса...")

        try:
            self._update_kb_files_from_terms_map()

            new_files, modified_files, unchanged_files = self._find_new_and_modified_files()

            self.stats["new_files"] = len(new_files)
            self.stats["modified_files"] = len(modified_files)

            self._cleanup_unchanged_files(unchanged_files)

            if not new_files and not modified_files:
                logger.info("Нет новых или изменённых файлов")
                self.stats["end_time"] = datetime.now()
                self.stats["total_chunks"] = len(self.chunks)
                if self.terms_map_updated:
                    self._save_terms_map()
                    self.terms_map_updated = False
                if self.fandom_pages_updated:
                    self._save_fandom_pages()
                    self.fandom_pages_updated = False
                self._save_update_log()
                return

            for filepath in modified_files:
                self._register_fandom_page(filepath)
                transformed_term = self._ensure_term_mapping(filepath.stem)
                doc_id = transformed_term.replace(' ', '_')
                transformed_filename = f"{doc_id}{filepath.suffix}"
                logger.info(f"Обработка изменённого файла: {filepath.name}")

                self._remove_old_chunks(doc_id)

                new_chunks, transformed_text = self._process_document(
                    filepath, doc_id, transformed_filename
                )

                if new_chunks:
                    self._add_new_chunks(new_chunks)

                    self.processed_files[filepath.name] = self._calculate_file_hash(filepath)

                    self._move_processed_file(
                        filepath, transformed_filename, transformed_text
                    )

            for filepath in new_files:
                self._register_fandom_page(filepath)
                transformed_term = self._ensure_term_mapping(filepath.stem)
                doc_id = transformed_term.replace(' ', '_')
                transformed_filename = f"{doc_id}{filepath.suffix}"
                logger.info(f"Обработка нового файла: {filepath.name}")

                new_chunks, transformed_text = self._process_document(
                    filepath, doc_id, transformed_filename
                )

                if new_chunks:
                    self._add_new_chunks(new_chunks)

                    self.processed_files[filepath.name] = self._calculate_file_hash(filepath)

                    self._move_processed_file(
                        filepath, transformed_filename, transformed_text
                    )

            self._save_index()

            self._save_processed_files()

            if self.terms_map_updated:
                self._save_terms_map()
                self.terms_map_updated = False
            if self.fandom_pages_updated:
                self._save_fandom_pages()
                self.fandom_pages_updated = False

            self.stats["end_time"] = datetime.now()
            self.stats["total_chunks"] = len(self.chunks)

            self._save_update_log()

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            self.stats["errors"].append(f"Critical: {str(e)}")
            self.stats["end_time"] = datetime.now()
            self._save_update_log()
            raise


if __name__ == "__main__":
    updater = IndexUpdater()
    updater.run()
