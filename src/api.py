#! /usr/bin/env python

from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_bot import RAGBot
from src.config import FEWSHOT_FILE
import json, os

app = FastAPI(title="RAG Bot")

bot = RAGBot()

fewshot_examples = []
if FEWSHOT_FILE and os.path.exists(FEWSHOT_FILE):
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            fewshot_examples.append(json.loads(line))

class Query(BaseModel):
    q: str
    use_fewshot: bool = True

@app.post("/ask")
def ask(q: Query):
    fs = fewshot_examples if q.use_fewshot else None
    resp = bot.answer(q.q)
    return resp

@app.get("/health")
def health():
    return {"status": "ok"}
