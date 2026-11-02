import os
import gc
import json
import time

# 1. HARD THREAD CAPS TO PREVENT CPU LOCKUPS (Must precede library imports)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch

# Limit PyTorch internal CPU threads
torch.set_num_threads(2)

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import ONNXReranker
from src.generator.llm import LocalGenerator

GOLDEN_DATASET = [
    {
        "question": "What role does S100 immunohistochemistry play in diagnosing salivary gland lesions?",
        "ground_truth": "S100 immunohistochemistry shows positivity in epithelial and myoepithelial cells, confirming the biphasic nature of lesions like sclerosing polycystic adenoma.",
        "key_terms": ["s100", "epithelial", "myoepithelial", "biphasic"]
    },
    {
        "question": "How are EGFR and CD117 receptors associated with histological grades in urothelial carcinoma?",
        "ground_truth": "Both EGFR and CD117 are individually and simultaneously associated with high histological grade in urothelial carcinomas.",
        "key_terms": ["egfr", "cd117", "histological grade", "urothelial"]
    },
    {
        "question": "What morphological subtypes are differentiated under triple-negative breast carcinoma?",
        "ground_truth": "Subtypes include secretory carcinoma, adenoid cystic carcinoma, tall cell carcinoma with reversed polarity, and metaplastic carcinoma.",
        "key_terms": ["secretory", "adenoid cystic", "tall cell", "metaplastic"]
    }
]

def run_evaluation():
    print("1/3 Loading Hybrid Retriever...")
    retriever = HybridRetriever()

    print("2/3 Loading ONNX Reranker...")
    reranker = ONNXReranker()

    print("3/3 Loading Local Generator (capped to 2 threads for thermal/OS safety)...")
    # 2 threads avoids saturating memory bus & prevents OS freezing
    generator = LocalGenerator(n_threads=2)

    # Re-use the retriever's encoder instead of loading a duplicate model into RAM
    judge_encoder = retriever.encoder

    results = []

    print("\n" + "=" * 80)
    print("STARTING STABLE LOCAL RAG BENCHMARK")
    print("=" * 80)

    for idx, item in enumerate(GOLDEN_DATASET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        key_terms = item["key_terms"]
        print(f"\n[Evaluating Sample {idx}/{len(GOLDEN_DATASET)}]: '{q}'")

        # 1. Retrieval + Re-ranking
        t0 = time.perf_counter()
        candidates = retriever.search(q, top_k=10)
        contexts = reranker.rerank(q, candidates, top_n=2)
        rerank_latency = (time.perf_counter() - t0) * 1000

        # Truncate text slices to prevent LLM prompt buffer overflow
        context_text_combined = " ".join([c["text"][:600] for c in contexts])

        # 2. Local Quantized Generation
        t1 = time.perf_counter()
        print("  Generating response on CPU (capped at 160 tokens)...", flush=True)
        answer = generator.generate(q, contexts, max_tokens=160)
        gen_latency = (time.perf_counter() - t1) * 1000

        # 3. Compute Metrics
        # Context Recall: Check keyword matches
        terms_retrieved = sum(1 for term in key_terms if term in context_text_combined.lower())
        context_recall_score = terms_retrieved / len(key_terms)

        # Answer Faithfulness: Embedding alignment
        ans_vec = judge_encoder.encode(answer, normalize_embeddings=True)
        ctx_vec = judge_encoder.encode(context_text_combined[:1500], normalize_embeddings=True)
        faithfulness_score = float(np.dot(ans_vec, ctx_vec))

        # Answer Relevancy
        gt_vec = judge_encoder.encode(gt, normalize_embeddings=True)
        relevancy_score = float(np.dot(ans_vec, gt_vec))

        results.append({
            "question": q,
            "faithfulness": round(faithfulness_score, 4),
            "answer_relevancy": round(relevancy_score, 4),
            "context_recall": round(context_recall_score, 4),
            "retrieval_ms": round(rerank_latency, 2),
            "generation_ms": round(gen_latency, 2)
        })

        print(f"  Sample {idx} complete: Faithfulness={faithfulness_score:.3f}, Recall={context_recall_score:.3f}")
        
        # Free temp buffers and sleep briefly to allow CPU core cooling
        gc.collect()
        time.sleep(1.0)

    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    output_csv = "data/ragas_benchmark_results.csv"
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df[["question", "faithfulness", "answer_relevancy", "context_recall", "generation_ms"]])
    print("\nAggregate Averages:")
    print(f"Mean Faithfulness:      {df['faithfulness'].mean():.4f}")
    print(f"Mean Answer Relevancy:  {df['answer_relevancy'].mean():.4f}")
    print(f"Mean Context Recall:    {df['context_recall'].mean():.4f}")
    print(f"Mean Generation Time:   {df['generation_ms'].mean() / 1000:.2f} s")

if __name__ == "__main__":
    run_evaluation()