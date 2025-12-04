# chunk_and_index.py
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from extract_text import load_file
from openai import OpenAI

client = OpenAI()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        if end < L:
            tail = text[start:end]
            br = max(tail.rfind("\n"), tail.rfind(" "))
            if br > chunk_size - 200:
                chunk = tail[:br]
                end = start + br
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    text = text[:20000]
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype=np.float32)

def build_index_from_file(file_path: str, out_dir: str = "index_data",
                          chunk_size: int = 1000, overlap: int = 200,
                          embed_model: str = "text-embedding-3-small"):
    os.makedirs(out_dir, exist_ok=True)
    text, ftype = load_file(file_path)
    print(f"Loaded {file_path} ({ftype}). Text length: {len(text)} chars")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Split into {len(chunks)} chunks")

    embeddings = []
    metadata = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks"), start=0):
        emb = get_embedding(chunk, model=embed_model)
        embeddings.append(emb)
        metadata.append({
            "id": i,
            "text": chunk,
            "source_file": os.path.basename(file_path)
        })

    embeddings = np.vstack(embeddings).astype(np.float32)

    # Try FAISS
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
        print("Saved FAISS index.")
    except Exception as e:
        print("FAISS unavailable or failed; saving numpy embeddings. Error:", e)
        np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    manifest = {"file_indexed": os.path.basename(file_path), "num_chunks": len(chunks), "embed_model": embed_model}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Index build complete. Saved to", out_dir)

# ===============================================
# NEW: Load FAISS index + metadata
# ===============================================
def load_faiss_index(index_dir: str = "index_data"):
    """
    Load FAISS index and metadata from disk.
    Returns (index, metadata) or (None, None) if not found.
    """
    import faiss

    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
