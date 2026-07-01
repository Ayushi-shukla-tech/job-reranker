#!/usr/bin/env python3
"""
One-time offline script: embed all 100k candidates with MiniLM → build FAISS index.
Run once before the main pipeline.

Usage:
    python scripts/precompute_embeddings.py --candidates <path_to_candidates.json>
"""
import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from data.loader import stream_candidates
from data.candidate_texts import build_minilm_text
from embeddings.minilm_encoder import MiniLMEncoder
from embeddings.faiss_index import FaissIndex

BATCH_SIZE = 512

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=settings.candidates_file)
    args = parser.parse_args()

    print(f"Loading MiniLM model...")
    encoder = MiniLMEncoder()
    index = FaissIndex()

    all_embeddings = []
    all_ids = []
    batch_texts = []
    batch_ids = []
    total = 0

    print(f"Streaming candidates from {args.candidates}...")
    for candidate in stream_candidates(args.candidates):
        text = build_minilm_text(candidate)
        batch_texts.append(text)
        batch_ids.append(candidate.candidate_id)

        if len(batch_texts) >= BATCH_SIZE:
            embs = encoder.encode(batch_texts)
            all_embeddings.append(embs)
            all_ids.extend(batch_ids)
            total += len(batch_ids)
            print(f"  Encoded {total} candidates...", end="\r")
            batch_texts, batch_ids = [], []

    if batch_texts:
        embs = encoder.encode(batch_texts)
        all_embeddings.append(embs)
        all_ids.extend(batch_ids)
        total += len(batch_ids)

    print(f"\nEncoded {total} candidates total.")
    print(f"Building FAISS index...")
    all_embs = np.vstack(all_embeddings)
    index.build(all_embs, all_ids)

    index_path = str(settings.embeddings_cache_dir / "minilm.index")
    index.save(index_path)

    ids_path = settings.embeddings_cache_dir / "candidate_ids.txt"
    ids_path.write_text("\n".join(all_ids))

    print(f"Done. Index saved to {index_path}.faiss / .ids")
    print(f"Candidate IDs saved to {ids_path}")
    print(f"Total: {total} candidates indexed.")

if __name__ == "__main__":
    main()
