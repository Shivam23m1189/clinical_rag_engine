import os
import pickle
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

BM25_INDEX_PATH = "data/bm25_index.pkl"
BM25_META_PATH = "data/metadata.pkl"
COLLECTION_NAME = "clinical_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        print("Initializing Hybrid Retriever...")
        
        # CPU SentenceTransformer for query embedding (1 query takes ~15ms on CPU)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.encoder.eval()
        
        # Qdrant client
        self.qdrant = QdrantClient(host="localhost", port=6333)
        
        # Load BM25 index and metadata
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
            
        with open(BM25_META_PATH, "rb") as f:
            self.metadata = pickle.load(f)
            
        print("Retriever ready.")

    def dense_search(self, query: str, top_k: int = 25) -> List[Dict]:
        query_vec = self.encoder.encode(query, normalize_embeddings=True).tolist()
        
        # Use query_points (or search)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True
        ).points
        
        hits = []
        for rank, hit in enumerate(results):
            payload = hit.payload
            hits.append({
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "title": payload.get("title", ""),
                "dense_rank": rank + 1,
                "score": hit.score
            })
        return hits

    def sparse_search(self, query: str, top_k: int = 25) -> List[Dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        hits = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            payload = self.metadata[idx]
            hits.append({
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "title": payload.get("title", ""),
                "sparse_rank": rank + 1,
                "score": float(scores[idx])
            })
        return hits

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        dense_results = self.dense_search(query, top_k=25)
        sparse_results = self.sparse_search(query, top_k=25)

        # Reciprocal Rank Fusion (RRF)
        # RRF_Score = sum( 1 / (k + rank) )
        fused_scores = {}
        doc_store = {}

        for hit in dense_results:
            cid = hit["chunk_id"]
            doc_store[cid] = hit
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + hit["dense_rank"]))

        for hit in sparse_results:
            cid = hit["chunk_id"]
            if cid not in doc_store:
                doc_store[cid] = hit
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + hit["sparse_rank"]))

        # Sort by RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]

        final_candidates = []
        for rank, cid in enumerate(sorted_cids):
            item = doc_store[cid]
            item["rrf_score"] = fused_scores[cid]
            item["final_rank"] = rank + 1
            final_candidates.append(item)

        return final_candidates

if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "What histological features identify invasive ductal carcinoma in epithelial biopsies?"
    
    print(f"\n--- Testing Query: '{test_query}' ---")
    results = retriever.search(test_query, top_k=3)
    
    for r in results:
        print(f"\n[Rank {r['final_rank']}] RRF Score: {r['rrf_score']:.5f} | Title: {r['title'][:60]}...")
        print(f"Excerpt: {r['text'][:250]}...")