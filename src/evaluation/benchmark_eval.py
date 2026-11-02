import os
import json
import time
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import ONNXReranker
from src.generator.llm import LocalGenerator

# Curated ground truth evaluation dataset
GOLDEN_EVAL_SET = [
    {
        "question": "What role does S100 immunohistochemistry play in diagnosing salivary gland lesions or adenomas?",
        "ground_truth": "S100 immunohistochemistry confirms the biphasic nature of lesions like sclerosing polycystic adenoma by demonstrating positivity in epithelial and myoepithelial cells.",
        "key_entities": ["s100", "biphasic", "epithelial", "myoepithelial", "adenoma"]
    },
    {
        "question": "How are EGFR and CD117 receptors evaluated in urothelial carcinoma?",
        "ground_truth": "EGFR and CD117 are evaluated using immunohistochemistry and are both individually and simultaneously associated with high histological grade in urothelial carcinomas.",
        "key_entities": ["egfr", "cd117", "immunohistochemistry", "histological grade"]
    },
    {
        "question": "What morphological criteria differentiate triple-negative breast carcinoma subtypes?",
        "ground_truth": "Morphological criteria include architectural, cytological, and stromal findings to differentiate subtypes like secretory carcinoma, adenoid cystic carcinoma, and metaplastic carcinoma.",
        "key_entities": ["secretory", "adenoid cystic", "metaplastic", "architectural"]
    },
    {
        "question": "What are the latest tax regulations published by SEBI for foreign institutional investors?",
        "ground_truth": "INSUFFICIENT_CONTEXT",
        "key_entities": []
    }
]

def calculate_precision_at_k(retrieved_contexts: list, key_entities: list) -> float:
    """Calculates Context Precision: ratio of retrieved chunks containing essential domain entities."""
    if not key_entities:
        return 1.0  # Out of domain query expectation
    hits = 0
    for ctx in retrieved_contexts:
        text_lower = ctx.lower()
        if any(ent in text_lower for ent in key_entities):
            hits += 1
    return hits / len(retrieved_contexts) if retrieved_contexts else 0.0

def run_evaluation():
    print("Initializing pipeline for evaluation...")
    retriever = HybridRetriever()
    reranker = ONNXReranker()
    generator = LocalGenerator(n_threads=4)
    eval_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    results = []

    print("\n" + "=" * 70)
    print("RUNNING OFFLINE QUANTITATIVE BENCHMARK")
    print("=" * 70)

    for idx, item in enumerate(GOLDEN_EVAL_SET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        entities = item["key_entities"]
        
        print(f"\nEvaluating Query [{idx}/{len(GOLDEN_EVAL_SET)}]: '{q}'")
        
        # 1. Pipeline Execution
        t0 = time.perf_counter()
        candidates = retriever.search(q, top_k=15)
        contexts = reranker.rerank(q, candidates, top_n=2)
        retrieval_time = (time.perf_counter() - t0) * 1000

        # Out-of-domain check
        if contexts and contexts[0].get("rerank_score", 0.0) < 0.15:
            generated_ans = "The provided clinical database does not contain sufficient or relevant evidence to answer this query safely."
        else:
            generated_ans = generator.generate(q, contexts, max_tokens=180)
            
        total_time = (time.perf_counter() - t0) * 1000
        context_texts = [c["text"] for c in contexts]

        # 2. Metric Calculations
        # Context Precision @ K
        ctx_precision = calculate_precision_at_k(context_texts, entities)

        if gt == "INSUFFICIENT_CONTEXT":
            # Negative rejection evaluation
            faithfulness_score = 1.0 if "insufficient" in generated_ans.lower() else 0.0
            semantic_similarity = 1.0 if faithfulness_score == 1.0 else 0.0
            rouge_l = 1.0 if faithfulness_score == 1.0 else 0.0
        else:
            # Semantic Similarity (Cosine similarity between answer and ground truth embedding)
            v_ans = eval_encoder.encode(generated_ans, normalize_embeddings=True)
            v_gt = eval_encoder.encode(gt, normalize_embeddings=True)
            semantic_similarity = float(np.dot(v_ans, v_gt))

            # ROUGE Scores
            scores = scorer.score(gt, generated_ans)
            rouge_l = scores["rougeL"].fmeasure

            # Faithfulness: Check claim coverage against retrieved contexts
            faithfulness_score = calculate_precision_at_k([generated_ans], entities)

        print(f"  -> Context Precision@2:  {ctx_precision:.3f}")
        print(f"  -> Semantic Alignment:   {semantic_similarity:.3f}")
        print(f"  -> ROUGE-L Score:        {rouge_l:.3f}")
        print(f"  -> Total E2E Latency:    {total_time:.1f} ms")

        results.append({
            "query": q,
            "generated_answer": generated_ans[:120] + "...",
            "context_precision": round(ctx_precision, 3),
            "semantic_similarity": round(semantic_similarity, 3),
            "rouge_l": round(rouge_l, 3),
            "latency_ms": round(total_time, 1)
        })

    os.makedirs("evaluation", exist_ok=True)
    df = pd.DataFrame(results)
    output_path = "evaluation/offline_eval_results.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE. Results exported to {output_path}")
    print("=" * 70)
    print(df[["query", "context_precision", "semantic_similarity", "rouge_l", "latency_ms"]])

if __name__ == "__main__":
    run_evaluation()