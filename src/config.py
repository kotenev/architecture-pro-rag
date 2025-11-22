#! /usr/bin/env python

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = os.environ.get("KB_DIR", str(BASE_DIR / "knowledge_base"))
INDEX_DIR = os.environ.get("INDEX_DIR", str(BASE_DIR / "index"))

EMBED_MODEL = "intfloat/Multilingual-E5-large"
EMBED_DIM = 1024
TOP_K = int(os.environ.get("TOP_K", 5))

YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_LLM_MODEL = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-5-lite")

FEWSHOT_FILE = os.environ.get("FEWSHOT_FILE", str(BASE_DIR / "examples" / "fewshot.jsonl"))
TERMS_MAP_FILE = os.environ.get("TERMS_MAP_FILE", str(BASE_DIR / "terms_map.json"))

SAFETY_BLOCKLIST = ["superpassword", "swordfish", "ignore all instructions", "root-password"]

def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


USE_EXTRACTIVE_FALLBACK_ON_LLM_UNKNOWN = _as_bool(
    os.environ.get("USE_EXTRACTIVE_FALLBACK_ON_LLM_UNKNOWN"), default=False
)
