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

### Reproducing Data Locally

To populate the local `data/` directory with the PMC chunks:

```bash
python data/download_data.py
```


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