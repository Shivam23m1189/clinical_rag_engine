import os
import json
import pickle
import torch
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Tune CPU threads for maximum core saturation without OS lockup
num_cores = os.cpu_count() or 4
torch.set_num_threads(4)

PROCESSED_CHUNKS_PATH = "data/processed_chunks.jsonl"
BM25_INDEX_PATH = "data/bm25_index.pkl"
BM25_METADATA_PATH = "data/bm25_metadata.pkl"

COLLECTION_NAME = "clinical_corpus"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384

# 64 items is the sweet spot for CPU cache efficiency
BATCH_SIZE = 64  
# Set MAX_CHUNKS to 35000 for a 15-minute run, or None for the full 148k
MAX_CHUNKS = 10000 

def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()

def main():
    print(f"Allocating {torch.get_num_threads()} CPU threads for inference.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    
    # Load model and switch to inference mode
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    model.eval()

    client = QdrantClient(host="localhost", port=6333)
    
    # Reset Qdrant Collection
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Collection '{COLLECTION_NAME}' exists. Resetting...")
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
        # On-disk payload storage prevents RAM saturation on 15GB Ubuntu host
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000)
    )

    corpus_tokens = []
    metadata_store = []
    
    batch_texts = []
    batch_payloads = []
    batch_ids = []
    current_point_id = 0

    print("Streaming and indexing chunks...")
    with open(PROCESSED_CHUNKS_PATH, "r", encoding="utf-8") as infile:
        with torch.inference_mode():
            for line in tqdm(infile, total=MAX_CHUNKS or 148017, desc="Indexing"):
                if not line.strip():
                    continue
                
                chunk = json.loads(line)
                text = chunk["text"]
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "pmid": chunk["pmid"],
                    "title": chunk["title"],
                    "text": text
                }
                
                corpus_tokens.append(simple_tokenize(text))
                metadata_store.append(payload)
                
                batch_texts.append(text)
                batch_payloads.append(payload)
                batch_ids.append(current_point_id)
                current_point_id += 1
                
                if len(batch_texts) >= BATCH_SIZE:
                    embeddings = model.encode(
                        batch_texts, 
                        batch_size=BATCH_SIZE, 
                        show_progress_bar=False, 
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=models.Batch(
                            ids=batch_ids,
                            vectors=embeddings.tolist(),
                            payloads=batch_payloads
                        )
                    )
                    batch_texts.clear()
                    batch_payloads.clear()
                    batch_ids.clear()
                    
                if MAX_CHUNKS and current_point_id >= MAX_CHUNKS:
                    print(f"Reached specified limit of {MAX_CHUNKS} chunks.")
                    break

        if batch_texts:
            embeddings = model.encode(
                batch_texts, 
                batch_size=len(batch_texts), 
                show_progress_bar=False, 
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=embeddings.tolist(),
                    payloads=batch_payloads
                )
            )

    print(f"\nIndexed {current_point_id} vectors into Qdrant.")

    print("Building BM25 sparse index...")
    bm25 = BM25Okapi(corpus_tokens)
    
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(BM25_METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"BM25 index successfully saved to {BM25_INDEX_PATH}")

if __name__ == "__main__":
    main()