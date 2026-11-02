# !pip install -q sentence-transformers rank_bm25

import os
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# UPDATE this path to where your file is located in your Drive:
# If in root of My Drive: '/content/drive/MyDrive/processed_chunks.jsonl'
# If inside a folder named 'RAG': '/content/drive/MyDrive/RAG/processed_chunks.jsonl'
DRIVE_INPUT_FILE = '/content/drive/MyDrive/rag_implementation/processed_chunks.jsonl'

# Local output paths on Colab VM (saved locally first for fast writing, then copied to Drive)
OUTPUT_VECTORS = "/content/drive/MyDrive/rag_implementation/vectors_384d.npy"
OUTPUT_BM25 = "/content/drive/MyDrive/rag_implementation/bm25_index.pkl"
OUTPUT_META = "/content/drive/MyDrive/rag_implementation/metadata.pkl"

if not os.path.exists(DRIVE_INPUT_FILE):
    raise FileNotFoundError(f"File not found at: {DRIVE_INPUT_FILE}. Check your Drive path.")

print("Loading model onto T4 GPU...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

texts = []
metadata = []
corpus_tokens = []

corrupted_lines = 0

print("Reading and parsing chunks from Google Drive...")
with open(DRIVE_INPUT_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(tqdm(f, desc="Parsing Lines")):
        clean_line = line.strip()
        if not clean_line:
            continue
        try:
            item = json.loads(clean_line)
            texts.append(item["text"])
            metadata.append({
                "chunk_id": item["chunk_id"],
                "pmid": item.get("pmid"),
                "title": item.get("title", ""),
                "text": item["text"]
            })
            corpus_tokens.append(item["text"].lower().split())
        except json.JSONDecodeError:
            corrupted_lines += 1
            continue

print(f"Successfully loaded {len(texts)} valid chunks. (Skipped corrupted lines: {corrupted_lines})")

# 2. Compute Dense Embeddings on GPU
print(f"Computing embeddings on GPU in batches of 512...")
embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}. Saving directly to Google Drive...")
np.save(OUTPUT_VECTORS, embeddings)

print("Saving metadata to Google Drive...")
with open(OUTPUT_META, "wb") as f:
    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Fit and Save BM25
print("Fitting BM25 sparse index...")
bm25 = BM25Okapi(corpus_tokens)

print("Saving BM25 index to Google Drive...")
with open(OUTPUT_BM25, "wb") as f:
    pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n--- Processing Complete ---")
print(f"Files saved directly to your Google Drive:")
print(f"1. {OUTPUT_VECTORS}")
print(f"2. {OUTPUT_META}")
print(f"3. {OUTPUT_BM25}")