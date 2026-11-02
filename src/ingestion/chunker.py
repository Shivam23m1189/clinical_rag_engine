import os
import json
import hashlib
from typing import List, Dict, Generator
import tiktoken
from tqdm import tqdm

DATA_PATH = "data/pmc_clinical_subset.jsonl"
OUTPUT_CHUNKS_PATH = "data/processed_chunks.jsonl"

CHUNK_SIZE = 400      # Target token count per chunk
CHUNK_OVERLAP = 50    # Overlap between consecutive chunks
MIN_CHUNK_TOKENS = 50 # Discard noisy/insignificant fragments

tokenizer = tiktoken.get_encoding("cl100k_base")

def get_text_hash(text: str) -> str:
    """SHA-256 hash to deduplicate exact duplicate paragraphs or articles."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def chunk_text_sliding_window(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Sliding window chunker operating directly on token space.
    Preserves context continuity and avoids mid-word truncations.
    """
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    chunks = []
    
    if total_tokens <= chunk_size:
        if total_tokens >= MIN_CHUNK_TOKENS:
            chunks.append({
                "chunk_text": text.strip(),
                "token_count": total_tokens
            })
        return chunks

    start = 0
    step = chunk_size - overlap
    
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_token_slice = tokens[start:end]
        
        if len(chunk_token_slice) >= MIN_CHUNK_TOKENS:
            chunk_str = tokenizer.decode(chunk_token_slice).strip()
            chunks.append({
                "chunk_text": chunk_str,
                "token_count": len(chunk_token_slice)
            })
        
        if end == total_tokens:
            break
        start += step

    return chunks

def process_corpus(input_path: str, output_path: str):
    seen_hashes = set()
    total_articles = 0
    total_chunks = 0
    duplicates_dropped = 0

    print(f"Reading from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in tqdm(infile, desc="Processing articles"):
            if not line.strip():
                continue
            
            record = json.loads(line)
            pmid = record.get("pmid")
            title = record.get("title", "Unknown Title")
            raw_text = record.get("text", "")
            
            # 1. Exact document deduplication
            doc_hash = get_text_hash(raw_text)
            if doc_hash in seen_hashes:
                duplicates_dropped += 1
                continue
            seen_hashes.add(doc_hash)
            total_articles += 1

            # 2. Token-level sliding window chunking
            chunks = chunk_text_sliding_window(raw_text)
            
            for idx, c in enumerate(chunks):
                chunk_entry = {
                    "chunk_id": f"{pmid or 'pmc'}_{idx}",
                    "pmid": pmid,
                    "title": title,
                    "text": c["chunk_text"],
                    "token_count": c["token_count"]
                }
                outfile.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("\n--- Ingestion Summary ---")
    print(f"Unique Articles Processed: {total_articles}")
    print(f"Duplicate Articles Dropped: {duplicates_dropped}")
    print(f"Total Chunks Generated:    {total_chunks}")
    print(f"Saved chunks to:           {output_path}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")
    process_corpus(DATA_PATH, OUTPUT_CHUNKS_PATH)