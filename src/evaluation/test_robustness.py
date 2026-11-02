import os
import sys
import time
from typing import List, Dict

# Prevent thread over-subscription across libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import ONNXReranker
from src.generator.llm import LocalGenerator

TEST_QUERIES = [
    {
        "category": "Direct Factual (In-Corpus Terminology)",
        "query": "What role does S100 immunohistochemistry play in diagnosing salivary gland lesions or adenomas?",
        "expected_behavior": "Should cite S100 positivity in epithelial/myoepithelial cells."
    },
    {
        "category": "Complex Multi-Aspect (Histopathology + Receptors)",
        "query": "How are EGFR and CD117 receptors evaluated in urothelial carcinoma?",
        "expected_behavior": "Should retrieve canine urothelial carcinoma paper and cite histological grading."
    },
    {
        "category": "Differential Diagnosis / Morphological Subtypes",
        "query": "What morphological criteria differentiate triple-negative breast carcinoma subtypes?",
        "expected_behavior": "Should cite cribriform, tubular, or high-grade patterns."
    },
    {
        "category": "Adversarial / Out-of-Domain (Negative Rejection)",
        "query": "What are the latest tax regulations published by SEBI for foreign institutional investors?",
        "expected_behavior": "Must cleanly reject: state context is insufficient."
    },
    {
        "category": "Ambiguous / Edge Case Query",
        "query": "What is the recommended dosage of metformin for treating acute glioblastoma multiforme?",
        "expected_behavior": "Should state context is insufficient."
    }
]

def run_evaluation_suite():
    print("Initializing pipeline components...")
    retriever = HybridRetriever()
    reranker = ONNXReranker()
    # Use 4 threads for optimal cache line sharing
    generator = LocalGenerator(n_threads=4)

    print("\n" + "=" * 80)
    print("STARTING OPTIMIZED ROBUSTNESS EVALUATION SUITE")
    print("=" * 80)

    for idx, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        category = test_case["category"]
        expected = test_case["expected_behavior"]

        print(f"\n[Test Case {idx}/{len(TEST_QUERIES)}]")
        print(f"Category:  {category}")
        print(f"Query:     '{query}'")
        print(f"Expect:    {expected}")
        print("-" * 60)

        # 1. Retrieval
        t0 = time.perf_counter()
        candidates = retriever.search(query, top_k=15)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # 2. Re-ranking: top 2 contexts is the sweet spot for CPU speed + accuracy
        t1 = time.perf_counter()
        contexts = reranker.rerank(query, candidates, top_n=2)
        rerank_ms = (time.perf_counter() - t1) * 1000

        print("\n[Retrieved Top Contexts]")
        for c_idx, doc in enumerate(contexts, 1):
            pmid = doc.get("pmid", "N/A")
            title = doc.get("title", "Unknown")[:60]
            score = doc.get("rerank_score", 0.0)
            print(f"  {c_idx}. [Score: {score:.3f}] PMID: {pmid} | Title: {title}...")

        # 3. Generation with visible heartbeat
        print("\n[Synthesizing Response (evaluating prompt on CPU...)]", flush=True)
        t2 = time.perf_counter()
        response_tokens = []
        first_token_received = False
        ttft_ms = 0.0

        for token in generator.stream_generate(query, contexts, max_tokens=220):
            if not first_token_received:
                first_token_received = True
                ttft_ms = (time.perf_counter() - t2) * 1000
                print(f" (TTFT: {ttft_ms:.0f}ms)\n> ", end="", flush=True)
            print(token, end="", flush=True)
            response_tokens.append(token)

        generation_ms = (time.perf_counter() - t2) * 1000
        full_answer = "".join(response_tokens)
        token_count = len(response_tokens)
        tps = token_count / (generation_ms / 1000) if generation_ms > 0 else 0

        # 4. Citation Inspection
        cited_sources = [
            doc.get("pmid")
            for doc in contexts
            if doc.get("pmid") and str(doc.get("pmid")) in full_answer
        ]

        print("\n\n[Performance Telemetry]")
        print(f"  - Hybrid Search Latency:   {retrieval_ms:.1f} ms")
        print(f"  - ONNX Rerank Latency:     {rerank_ms:.1f} ms")
        print(f"  - Time to First Token:     {ttft_ms:.1f} ms")
        print(f"  - Generation Throughput:   {tps:.1f} tokens/sec ({token_count} tokens in {generation_ms/1000:.2f}s)")
        print(f"  - Verified Cited PMIDs:    {cited_sources if cited_sources else 'None cited / Context insufficient'}")
        print("=" * 80)

if __name__ == "__main__":
    run_evaluation_suite()