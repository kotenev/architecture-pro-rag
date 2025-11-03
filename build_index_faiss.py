#! /usr/bin/env python

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("--- Начало процесса создания индекса FAISS ---")

loader_kwargs = {'encoding': 'utf-8'}
loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
    loader_kwargs=loader_kwargs
)

documents = loader.load()
if not documents:
    print(f"Ошибка: Не найдены документы в директории {KNOWLEDGE_BASE_DIR}")
    exit()

print(f"Загружено {len(documents)} документов.")

print("Добавление метаданных (имя файла)...")
for doc in documents:
    doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Документы разбиты на {len(chunks)} чанков.")

print("Добавление ID для каждого чанка...")
for i, chunk in enumerate(chunks):
    chunk.metadata['chunk_id'] = f"{chunk.metadata['source']}-{i}"

print(f"Загрузка эмбеддинг-модели: {EMBEDDING_MODEL_NAME}...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Модель успешно загружена.")

print("Создание FAISS индекса в памяти... (это может занять время)")
db = FAISS.from_documents(chunks, embedding_model)

print(f"Сохранение индекса на диск в директорию: {FAISS_INDEX_DIR}...")
db.save_local(FAISS_INDEX_DIR)

print("--- Процесс создания индекса FAISS успешно завершен! ---")
print(f"Индекс сохранен в папке: {FAISS_INDEX_DIR}")
