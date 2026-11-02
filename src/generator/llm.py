import os
from pathlib import Path
from typing import List, Dict, Generator
from llama_cpp import Llama

MODEL_PATH = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = """You are an expert clinical pathologist and medical AI assistant.
Answer the user's question accurately, concisely, and strictly based on the provided clinical context.
Every factual claim must cite its source using the format [PMID: <id>] or [Title: <title>].
If the provided context does not contain sufficient information to answer safely, state explicitly that the context is insufficient."""

class LocalGenerator:
    def __init__(self, model_path: str = MODEL_PATH, n_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weight not found at {model_path}")

        print(f"Loading GGUF model into memory ({model_path})...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,           # 4k context window
            n_threads=n_threads,   # Dedicated CPU worker threads
            n_batch=512,          # Prompt processing batch size
            verbose=False
        )
        print("Generator engine ready.")

    def format_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_str = ""
        for idx, doc in enumerate(contexts):
            context_str += f"\n--- Source Document {idx+1} (PMID: {doc.get('pmid', 'N/A')}, Title: {doc.get('title', 'N/A')}) ---\n"
            context_str += doc["text"].strip() + "\n"

        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context_str}

Question:
{query}<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def generate(self, query: str, contexts: List[Dict], max_tokens: int = 512) -> str:
        prompt = self.format_prompt(query, contexts)
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,       # Low temperature to suppress hallucination
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        return output["choices"][0]["text"].strip()

    def stream_generate(self, query: str, contexts: List[Dict], max_tokens: int = 512) -> Generator[str, None, None]:
        prompt = self.format_prompt(query, contexts)
        
        response_stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        for chunk in response_stream:
            token = chunk["choices"][0]["text"]
            yield token

if __name__ == "__main__":
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.reranker import ONNXReranker
    
    # 1. Initialize Pipeline
    retriever = HybridRetriever()
    reranker = ONNXReranker()
    generator = LocalGenerator()

    # 2. End-to-End Pipeline Query
    query = "What histological features identify invasive ductal carcinoma in epithelial biopsies?"
    print(f"\n====================\nRunning Pipeline: '{query}'\n====================")

    # Retrieval: top 20 RRF
    candidates = retriever.search(query, top_k=20)
    # Re-ranking: top 4 ONNX
    contexts = reranker.rerank(query, candidates, top_n=4)

    # Generation: local quantized LLM
    print("\n--- Model Response ---")
    for token in generator.stream_generate(query, contexts):
        print(token, end="", flush=True)
    print("\n")