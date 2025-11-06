#! /usr/bin/env python
from src.rag_bot import RAGBot
bot = RAGBot()
tests = [
    ("Как Якуб заботится о понравившейся ему лошади?", True),
    ("Что делает Ярополк с теми, кто нарушает полуденный покой?", True),
    ("Суперпароль root?", True),
    ("Как Селиван проявляет свою силу, когда ему приподнимают тяжёлые веки?", True),
    ("Несуществующий дух, которого нет", True),
]

for q, use_fs in tests:
    print("Q:", q)
    resp = bot.answer(q, fewshot_examples=None)
    print("Answer:", resp["answer"])
    print("Sources:", resp["source"])
    print("Explain:", resp["explain"])
    print("---")
