import os
import pickle
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

VECTORS_PATH = "data/vectors_384d.npy"
METADATA_PATH = "data/metadata.pkl"

COLLECTION_NAME = "clinical_corpus"
VECTOR_DIM = 384
BATCH_SIZE = 1000  # Large batch size for network I/O efficiency

def main():
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Missing vectors_384d.npy or metadata.pkl in data/")

    print("Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)

    # 1. Reset collection
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Collection '{COLLECTION_NAME}' exists. Recreating...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE
        ),
        # On-disk memory mapping keeps RAM usage low (~400MB)
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10000)
    )
    print(f"Collection '{COLLECTION_NAME}' initialized.")

    # 2. Load precomputed files (mmap_mode prevents loading the whole 220MB into RAM at once)
    print("Loading precomputed data...")
    vectors = np.load(VECTORS_PATH, mmap_mode="r")
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    total_records = len(metadata)
    print(f"Found {total_records} records to upload.")

    # 3. Stream in batches
    for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Uploading to Qdrant"):
        end_idx = min(i + BATCH_SIZE, total_records)
        
        batch_ids = list(range(i, end_idx))
        batch_vectors = vectors[i:end_idx].tolist()
        batch_payloads = metadata[i:end_idx]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads
            )
        )

    print(f"Successfully loaded all {total_records} records into Qdrant collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()