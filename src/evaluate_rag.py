#! /usr/bin/env python
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.rag_bot import RAGBot
from src.config import INDEX_DIR, FEWSHOT_FILE, BASE_DIR
from src.logging_utils import log_interaction, compute_success_flag


GOLDEN_FILE = BASE_DIR / "tests" / "golden_questions.jsonl"


def load_golden_questions(path: Path) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def run_single_test(bot: RAGBot, item: Dict[str, Any]) -> Tuple[str, bool]:
    qid = str(item.get("id"))
    query = str(item.get("q", "")).strip()
    expected = item.get("expected")

    print(f"[TEST] {qid}: {query} (expected={expected})")

    resp = bot.answer(query)

    log_interaction(
        query=query,
        answer_dict=resp,
        golden_id=qid,
        expected=expected,
    )

    answer_text = str(resp.get("answer", "")).strip()
    sources_raw = resp.get("source")

    if sources_raw is None:
        sources = []
    elif isinstance(sources_raw, str):
        sources = [sources_raw]
    else:
        try:
            sources = list(sources_raw)
        except TypeError:
            sources = [str(sources_raw)]

    ok_flag = compute_success_flag(
        answer_text=answer_text,
        sources=sources,
        expected=expected,
    )

    ok = bool(ok_flag)
    status = "OK" if ok else "FAIL"
    print(f" -> {status}")
    if not ok:
        print("    Ответ:", answer_text[:200].replace("\n", " "))

    return qid, ok


def main() -> None:
    if not GOLDEN_FILE.exists():
        raise SystemExit(f"Не найден файл с золотыми вопросами: {GOLDEN_FILE}")

    questions = load_golden_questions(GOLDEN_FILE)
    if not questions:
        raise SystemExit("Файл golden_questions.jsonl пуст или невалиден.")

    print(f"Загружено золотых вопросов: {len(questions)}")
    print(f"INDEX_DIR = {INDEX_DIR}")
    print(f"FEWSHOT_FILE = {FEWSHOT_FILE}")
    print()

    bot = RAGBot(
        index_dir=str(INDEX_DIR),
        fewshot_file=str(FEWSHOT_FILE),
        use_llm=True,
    )

    results: List[Tuple[str, bool]] = []

    for item in questions:
        qid, ok = run_single_test(bot, item)
        results.append((qid, ok))

    total = len(results)
    ok_count = sum(1 for _, ok in results if ok)
    fail = [(qid, ok) for qid, ok in results if not ok]

    print("\n=== Итоги оценки ===")
    print(f"Всего вопросов: {total}")
    print(f"Успешных ответов: {ok_count}")
    print(f"Провалов: {total - ok_count}")

    if fail:
        print("\nНеуспешные кейсы:")
        for qid, _ in fail:
            print(f" - {qid}")


if __name__ == "__main__":
    main()
