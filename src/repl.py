#! /usr/bin/env python

from src.rag_bot import RAGBot
import json, sys
bot = RAGBot()

fewshot = [
    {"q":"Как называется столица планеты Ти’лора?", "a":"Столица планеты Ти’лора называется Сайрон."},
    {"q":"Кто создал ядро VoidCore?", "a":"Ядро VoidCore было разработано командой VoidLabs."}
]

def main():
    print("RAG REPL. Введите 'exit' для выхода.")
    while True:
        q = input("Q: ").strip()
        if q in ("exit", "quit"): break
        use_fs = input("Использовать few-shot? (y/n) ").lower().startswith("y")
        resp = bot.answer(q, fewshot_examples=fewshot if use_fs else None)
        print("---\nANSWER:\n", resp["answer"])
        print("SOURCES:", resp["source"])
        print("EXPLAIN:", resp["explain"])
        print("-----\n")

if __name__ == "__main__":
    main()
