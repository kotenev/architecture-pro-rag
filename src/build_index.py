#! /usr/bin/env python
"""
Запуск:
python src/build_index.py --kb-dir ../knowledge_base --index-dir ../index
"""
import argparse, os, json, math
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
from src.utils import load_txt_files, save_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, p in enumerate(pieces):
            chunks.append({
                "source_id": doc["id"],
                "chunk_id": f"{doc['id']}_{i}",
                "text": p
            })
    return chunks

def build_index(kb_dir, index_dir, embed_model_name):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embed_model_name)
    docs = load_txt_files(kb_dir)
    print(f"Loaded {len(docs)} documents from {kb_dir}")

    chunks = chunk_texts(docs, chunk_size=500, chunk_overlap=50)
    print(f"Generated {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # save metadata
    metadata = {"chunks": chunks, "embed_dim": dim, "model": embed_model_name}
    save_metadata(metadata, str(index_dir / "metadata.json"))
    print("Index built and saved to", index_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", default="../knowledge_base")
    parser.add_argument("--index-dir", default="../index")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    build_index(args.kb_dir, args.index_dir, args.embed_model)
