import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

QUANTIZED_MODEL_DIR = Path("models/onnx_reranker_quantized")

class ONNXReranker:
    def __init__(self, model_dir: Path = QUANTIZED_MODEL_DIR):
        print("Loading Quantized ONNX Cross-Encoder onto CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Configure ONNX Runtime session for multi-threaded CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Matches optimal CPU cache cores
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        onnx_model_path = model_dir / "model_quantized.onnx"
        if not onnx_model_path.exists():
            onnx_model_path = model_dir / "model.onnx"
            
        self.session = ort.InferenceSession(
            str(onnx_model_path), 
            sess_options, 
            providers=["CPUExecutionProvider"]
        )
        print("ONNX Reranker session initialized.")

    def compute_score(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        # Tokenize pairs: [[query, doc_1], [query, doc_2], ...]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        
        # Execute ONNX forward pass
        ort_outputs = self.session.run(None, ort_inputs)
        # Logits output for sequence classification
        scores = ort_outputs[0].squeeze(-1)
        
        # Apply sigmoid to normalize to [0, 1] range
        scores = 1.0 / (1.0 + np.exp(-scores))
        return scores

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 4) -> List[Dict]:
        if not candidates:
            return []

        start_time = time.perf_counter()
        pairs = [(query, c["text"]) for c in candidates]
        
        scores = self.compute_score(pairs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # Sort by cross-encoder score descending
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
        
        for rank, item in enumerate(reranked):
            item["rerank_rank"] = rank + 1

        print(f"Reranked {len(candidates)} candidates in {latency_ms:.2f} ms")
        return reranked

if __name__ == "__main__":
    from src.retrieval.hybrid_search import HybridRetriever
    
    retriever = HybridRetriever()
    reranker = ONNXReranker()
    
    query = "What histological features identify invasive ductal carcinoma in epithelial biopsies?"
    
    # 1. Retrieve top 20 candidates via Hybrid RRF
    candidates = retriever.search(query, top_k=20)
    
    # 2. Re-rank top 20 down to top 4 using ONNX cross-encoder
    top_contexts = reranker.rerank(query, candidates, top_n=4)
    
    print("\n--- Final Re-Ranked Results ---")
    for r in top_contexts:
        print(f"\n[Rank {r['rerank_rank']}] Score: {r['rerank_score']:.4f} (Original RRF Rank: {r['final_rank']})")
        print(f"Title: {r['title']}")
        print(f"Snippet: {r['text'][:250]}...")