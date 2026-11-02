# Edge-Optimized Clinical RAG Platform (148k PMC Pathology Corpus)

An end-to-end, locally hosted Clinical Retrieval-Augmented Generation (RAG) platform optimized for resource-constrained, CPU-only hardware (15 GB RAM, no dedicated GPU). 

The system indexes over **148,000 biomedical chunks** extracted from a curated subset of 5,000 open-access pathology papers in PubMed Central (PMC), providing grounded histopathology insights with zero external API dependencies.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([User Query]) --> Cache[Redis Semantic Cache]

    %% Cache Hit Path
    Cache -- "Cosine Sim ≥ 0.94<br>(Cache Hit)" --> FastReturn[Return Cached Response<br>&lt; 25 ms]

    %% Cache Miss Path
    Cache -- "Cosine Sim < 0.94<br>(Cache Miss)" --> Retrieval

    %% Hybrid Retrieval Subgraph
    subgraph Retrieval [1. Two-Stage Hybrid Retrieval Layer]
        direction TB
        Dense["Dense Vector Search<br>• Qdrant HNSW Index<br>• BAAI/bge-small-en-v1.5 (384-d)"]
        Sparse["Sparse Keyword Search<br>• BM25Okapi Index<br>• Sub-word / Token Matching"]
        RRF["Rank Fusion<br>• Reciprocal Rank Fusion (RRF, k=60)"]

        Dense --> RRF
        Sparse --> RRF
    end

    Retrieval -->|Top-15 Fused Candidates| Reranker

    %% Cross-Encoder Subgraph
    subgraph Reranker [2. Cross-Encoder Re-Ranking Layer]
        direction TB
        ONNX["BAAI/bge-reranker-base<br>• Quantized INT8 ONNX Runtime<br>• Multi-Threaded AVX2 CPU Inference"]
        Filter{"Relevance Gate<br>Score &lt; 0.15?"}

        ONNX --> Filter
    end

    %% Rejection vs Generation
    Filter -- "Yes (Adversarial / Out-of-Domain)" --> Reject[Negative Rejection Fallback<br>Safe Inconclusive Response]
    Filter -- "No (Top-2 Grounded Contexts)" --> Generator

    %% Generation Subgraph
    subgraph Generator [3. Local Quantized Generation Layer]
        direction TB
        LLM["Qwen2.5-3B-Instruct<br>• GGUF 4-bit (Q4_K_M)<br>• llama.cpp Engine (Thread-Capped)"]
    end

    Generator --> SSE([Streamed Token Response<br>FastAPI SSE])
    Generator -.->|Write-Back to Cache| Cache

    %% Styling
    style User fill:#1f2937,stroke:#9ca3af,stroke-width:2px,color:#fff
    style FastReturn fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style Reject fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff
    style SSE fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Retrieval fill:#111827,stroke:#374151,color:#e5e7eb
    style Reranker fill:#111827,stroke:#374151,color:#e5e7eb
    style Generator fill:#111827,stroke:#374151,color:#e5e7eb
```

## 🚀 Key Engineering Highlights

1. **Two-Stage Hybrid Search & RRF:** Combines dense cosine similarity (`BAAI/bge-small-en-v1.5`) with exact token matching (`BM25Okapi`). Rank-based Reciprocal Rank Fusion ($k=60$) resolves score-scale discrepancies between sparse and dense retrievers without heuristic weight tuning.
2. **Quantized ONNX Cross-Encoder Re-Ranking:** Directly exported and dynamically quantized `BAAI/bge-reranker-base` to INT8 ONNX Runtime with AVX2 vector extensions. Scores 10–15 candidate pairs in **~2.3 seconds on CPU**, promoting high-signal passages up to 15 positions higher than raw bi-encoder search.
3. **Redis Semantic Caching:** Pre-computes normalized query embeddings to evaluate incoming query similarity against a $0.94$ cosine threshold. Drops repeat and rephrased query latencies from **46.8s to 21.3ms (~2,190x speedup)**.
4. **Thermal & Memory Constraints:** Thread-capped workers (`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`) and on-disk memory mapping (`memmap_threshold=10000` in Qdrant) maintain execution under a **4.5 GB RAM footprint**, eliminating system freezes and OOM crashes.
5. **Grounded Negative Rejection:** Employs calibrated cross-encoder gating (rejection threshold $< 0.15$) and clinical prompt constraints to reject out-of-domain queries (e.g., non-clinical tax/legal queries) and prevent off-label hallucination.

## 📊 Benchmark Metrics (Local Evaluation)

Quantitatively evaluated against a curated clinical test set derived from PMC pathology papers:

  

| **Metric**                 | **Score**   | **Description**                                                           |
| -------------------------- | ----------- | ------------------------------------------------------------------------- |
| **Mean Faithfulness**      | **0.8743**  | Measures semantic consistency between generated answer and source context |
| **Mean Answer Relevancy**  | **0.8582**  | Semantic accuracy against expert pathologist ground truth                 |
| **Mean Context Recall**    | **0.7500**  | Ground-truth key term coverage within top-2 retrieved contexts            |
| **ONNX Re-rank Latency**   | **~2.3 s**  | Quantized cross-encoder scoring latency for 10 candidate pairs on CPU     |
| **Semantic Cache Latency** | **21.3 ms** | Response latency on semantic cache hits ($Sim \ge 0.94$)                  |



Dataset: The PubMed Central Open Access dataset contains millions of full-text papers; we will use a high-quality clinical subset of 5000 pathology papers for this project.

##  Project Structure & Data Pipeline

Large binary artifacts (embeddings, vector indices, raw corpora) are excluded from version control via `.gitignore` to keep the repository lightweight. A reproducible data script is included to generate local assets.

```text
clinical_rag_engine/
├── data/
│   ├── download_data.py          # Tracked: Pulls & parses open-access PMC dataset
│   ├── processed_chunks.jsonl    # Ignored: 148k processed text chunks
│   ├── vectors_384d.npy          # Ignored: Precomputed 384-d dense embeddings
│   ├── bm25_index.pkl            # Ignored: Pickled sparse BM25 token index
│   └── metadata.pkl              # Ignored: Chunk metadata lookup store
├── models/                       # Ignored: GGUF weights & quantized ONNX models
├── qdrant_storage/               # Ignored: Local Qdrant HNSW vector persistence
├── src/                          # Core RAG source code
│   ├── api/                      # FastAPI service & Redis semantic cache
│   ├── evaluation/               # Robustness & benchmark test harnesses
│   ├── generator/                # llama.cpp local quantized inference
│   └── retrieval/                # Hybrid search (Dense + BM25 + RRF) & ONNX reranker
├── .gitignore
├── README.md
└── requirements.txt
```

## 🛠️ Tech Stack

- **Vector Database:** Qdrant (Dockerized, Cosine Metric, Memory-mapped on-disk storage)
- **Keyword Index:** Rank-BM25 (Sparse inverted index)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors)
- **Reranker Engine:** `BAAI/bge-reranker-base` (ONNX Runtime, Dynamic INT8 Quantization)
- **LLM Engine:** `Qwen2.5-3B-Instruct` (GGUF Q4_K_M via `llama-cpp-python`)
- **Caching & API:** Redis 7 (Docker), FastAPI, Uvicorn, Server-Sent Events (SSE)
- **Benchmarking:** Custom local quantitative evaluation suite

## 🚦 Quickstart & Local Reproduction

### 1. Prerequisites & Container Initialization

Ensure Docker is installed and running:
```bash
# Start Qdrant vector store with memory constraints
docker run -d --name qdrant-rag \
  --restart unless-stopped \
  -m 2g \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Start Redis semantic cache
docker run -d --name redis-rag \
  --restart unless-stopped \
  -p 6379:6379 \
  redis:alpine
```

### 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Data Processing & Ingestion

```bash
# 1. Download and parse PMC corpus
python data/download_data.py

# 2. Ingest dense embeddings into Qdrant
python -m src.retrieval.load_to_qdrant

# 3. Export and quantize ONNX reranker
python -m src.retrieval.export_reranker_onnx
```

### 4. Run the Production API

```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 5. Test Query & Streaming

```bash
# Standard REST endpoint (evaluates cache -> hybrid retrieval -> rerank -> LLM)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How are EGFR and CD117 receptors evaluated in urothelial carcinoma?"}'

# Real-time token stream via Server-Sent Events (SSE)
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the morphological criteria for triple-negative breast carcinoma subtypes?"}'
```

### 6. Run Quantitative Benchmarks

```bash
python -m src.evaluation.run_ragas_eval
```
