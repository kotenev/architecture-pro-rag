#! /usr/bin/env python

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = os.environ.get("KB_DIR", str(BASE_DIR / "knowledge_base"))
INDEX_DIR = os.environ.get("INDEX_DIR", str(BASE_DIR / "index"))

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = int(os.environ.get("TOP_K", 5))

YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_LLM_MODEL = os.environ.get("YANDEX_LLM_MODEL", "yandexgpt-lite")

FEWSHOT_FILE = os.environ.get("FEWSHOT_FILE", str(BASE_DIR / "examples" / "fewshot.jsonl"))
TERMS_MAP_FILE = os.environ.get("TERMS_MAP_FILE", str(BASE_DIR / "terms_map.json"))

SAFETY_BLOCKLIST = ["superpassword", "swordfish", "ignore all instructions", "root-password"]
