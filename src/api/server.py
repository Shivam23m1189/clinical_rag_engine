import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import ONNXReranker
from src.generator.llm import LocalGenerator
from src.api.cache import SemanticCache

pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Spinning up production inference engines...")
    pipeline["retriever"] = HybridRetriever()
    pipeline["reranker"] = ONNXReranker()
    pipeline["generator"] = LocalGenerator(n_threads=4)
    pipeline["cache"] = SemanticCache()
    print("All engines online.")
    yield
    pipeline.clear()

app = FastAPI(title="Clinical RAG Production Engine", version="1.0.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    top_contexts: int = 2

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "clinical-rag-engine"}

@app.post("/query")
async def query_pipeline(req: QueryRequest):
    start_time = time.perf_counter()
    query = req.query.strip()
    cache = pipeline["cache"]
    retriever = pipeline["retriever"]
    reranker = pipeline["reranker"]
    generator = pipeline["generator"]

    # 1. Check Semantic Cache
    cached_hit = cache.lookup(query)
    if cached_hit:
        return {
            "cached": True,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "similarity": cached_hit["similarity"],
            "response": cached_hit["answer"],
            "sources": cached_hit["sources"]
        }

    # 2. Hybrid Search (Dense + BM25)
    candidates = retriever.search(query, top_k=15)

    # 3. Dynamic Filtering via ONNX Cross-Encoder
    contexts = reranker.rerank(query, candidates, top_n=req.top_contexts)

    # Reject out-of-domain queries if cross-encoder score is negligible
    if not contexts or contexts[0].get("rerank_score", 0.0) < 0.15:
        response_text = "The provided clinical database does not contain sufficient or relevant evidence to answer this query safely."
        sources = []
    else:
        # 4. Generate Grounded Response
        response_text = generator.generate(query, contexts, max_tokens=220)
        sources = [
            {
                "title": c.get("title", "Unknown"),
                "pmid": c.get("pmid"),
                "relevance_score": round(c.get("rerank_score", 0.0), 4)
            }
            for c in contexts
        ]

    # Store in semantic cache
    cache.store(query, response_text, sources)

    total_latency = (time.perf_counter() - start_time) * 1000
    return {
        "cached": False,
        "latency_ms": round(total_latency, 2),
        "response": response_text,
        "sources": sources
    }

@app.post("/stream")
async def stream_pipeline(req: QueryRequest):
    query = req.query.strip()
    cache = pipeline["cache"]
    retriever = pipeline["retriever"]
    reranker = pipeline["reranker"]
    generator = pipeline["generator"]

    cached_hit = cache.lookup(query)
    if cached_hit:
        def stream_cached():
            yield f"data: {json.dumps({'token': cached_hit['answer'], 'cached': True})}\n\n"
        return StreamingResponse(stream_cached(), media_type="text/event-stream")

    candidates = retriever.search(query, top_k=15)
    contexts = reranker.rerank(query, candidates, top_n=req.top_contexts)

    def event_generator():
        collected_tokens = []
        for token in generator.stream_generate(query, contexts, max_tokens=220):
            collected_tokens.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        full_text = "".join(collected_tokens)
        sources = [{"title": c.get("title"), "pmid": c.get("pmid")} for c in contexts]
        cache.store(query, full_text, sources)

    return StreamingResponse(event_generator(), media_type="text/event-stream")