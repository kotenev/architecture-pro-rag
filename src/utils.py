import os, json
from pathlib import Path

def load_txt_files(folder):
    folder = Path(folder)
    docs = []
    for p in sorted(folder.glob("*.txt")):
        docs.append({"id": p.stem, "text": p.read_text(encoding="utf-8"), "path": str(p)})
    return docs

def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
