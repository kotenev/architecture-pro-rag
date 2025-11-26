#! /usr/bin/env python
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import BASE_DIR

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "logs.jsonl"


def _normalize_sources(raw_sources: Any) -> List[str]:
    if raw_sources is None:
        return []
    if isinstance(raw_sources, str):
        return [raw_sources]
    if isinstance(raw_sources, Sequence):
        return [str(s) for s in raw_sources]
    return [str(raw_sources)]


def compute_success_flag(
    *,
    answer_text: str,
    sources: List[str],
    expected: Optional[str],
) -> Optional[bool]:

    if expected not in {"known", "unknown"}:
        return None

    lower = answer_text.lower()
    chunks_found = bool(sources)
    long_enough = len(answer_text) >= 80
    says_dont_know = "я не знаю" in lower

    if expected == "known":
        return chunks_found and long_enough and not says_dont_know

    return (not chunks_found) or says_dont_know


def log_interaction(
    *,
    query: str,
    answer_dict: Dict[str, Any],
    log_file: Path = DEFAULT_LOG_FILE,
    golden_id: Optional[str] = None,
    expected: Optional[str] = None,
) -> None:

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    answer_text = str(answer_dict.get("answer", "")).strip()
    sources = _normalize_sources(answer_dict.get("source"))

    record: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer_text,
        "answer_length": len(answer_text),
        "chunks_found": bool(sources),
        "sources": sources,
    }

    if expected is not None:
        record["expected"] = expected
    if golden_id is not None:
        record["golden_id"] = golden_id

    success = compute_success_flag(
        answer_text=answer_text,
        sources=sources,
        expected=expected,
    )
    if success is not None:
        record["success"] = success

    with log_file.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
