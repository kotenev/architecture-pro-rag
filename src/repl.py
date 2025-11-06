#! /usr/bin/env python

from src.rag_bot import RAGBot
import json, sys
bot = RAGBot()

fewshot = [
    {"q": "Кто такой Якуб в новой базе знаний и как он проявляет заботу о лошади, которая ему по нраву?",
     "a": "Якуб описан как домашний дух-хранитель: он живёт у печи, следит за хозяйством и, если ему нравится лошадь хозяев, холит её, заплетает гриву и хвост и обеспечивает хорошим кормом."},
    {
        "q": "Что, согласно материалам о Ярополке, происходит с теми, кто нарушает полуденный отдых и сталкивается с ним на поле?",
        "a": "Ярополк предстает духом полуденного зноя: он преследует тех, кто работает в поле в полдень, может заставить разгадывать загадки, защекотать до изнеможения, навеять смертельный сон и даже отрезать голову упорным нарушителям."}
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
