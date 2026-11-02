import json
from typing import Optional, Dict
import numpy as np
import redis
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, host: str = "localhost", port: int = 6379, threshold: float = 0.94):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.threshold = threshold
        # Re-use the lightweight BGE query encoder (~400MB)
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        print("Redis Semantic Cache initialized.")

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.encoder.encode(text, normalize_embeddings=True)

    def lookup(self, query: str) -> Optional[Dict]:
        query_vec = self._get_embedding(query)
        keys = self.r.keys("cache:*")

        for k in keys:
            raw_data = self.r.get(k)
            if not raw_data:
                continue
            cached_data = json.loads(raw_data)
            cached_vec = np.array(cached_data["vector"])
            # Normalized vectors: cosine similarity == dot product
            sim = float(np.dot(query_vec, cached_vec))
            if sim >= self.threshold:
                print(f"[Cache Hit] Sim: {sim:.4f} matched: '{cached_data['query']}'")
                return {
                    "answer": cached_data["answer"],
                    "sources": cached_data["sources"],
                    "similarity": sim
                }
        return None

    def store(self, query: str, answer: str, sources: list):
        query_vec = self._get_embedding(query).tolist()
        cache_id = f"cache:{abs(hash(query))}"
        payload = {
            "query": query,
            "vector": query_vec,
            "answer": answer,
            "sources": sources
        }
        # TTL of 24 hours
        self.r.setex(cache_id, 86400, json.dumps(payload))