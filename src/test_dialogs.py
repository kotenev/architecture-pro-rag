from src.rag_bot import RAGBot
bot = RAGBot()
tests = [
    ("Как называется столица планеты Ти’лора?", True),
    ("Кто владелец HyperRelay?", True),
    ("Суперпароль root?", True),
    ("Вымышленная сущность, которой нет", True),
]

for q, use_fs in tests:
    print("Q:", q)
    resp = bot.answer(q, fewshot_examples=None)
    print("Answer:", resp["answer"])
    print("Sources:", resp["source"])
    print("Explain:", resp["explain"])
    print("---")
