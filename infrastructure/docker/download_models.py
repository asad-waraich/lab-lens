#!/usr/bin/env python3
"""
Script to pre-download embedding models during Docker build
"""
import os
import sys

cache_dir = "/root/.cache/huggingface"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

print(f"Cache directory: {cache_dir}")
print(f"Cache directory exists: {os.path.exists(cache_dir)}")
print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")

from sentence_transformers import SentenceTransformer

# Download default model
print("Downloading all-MiniLM-L6-v2...")
try:
    # Explicitly set device to CPU to avoid meta tensor issues
    import torch

    device = "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir, device=device)
    print("Model loaded successfully")

    # Force model to CPU and ensure it's not in meta state
    if hasattr(model, "_modules"):
        for module in model._modules.values():
            if hasattr(module, "to"):
                module.to("cpu")

    print("Verifying model can encode...")
    test_embedding = model.encode("test", convert_to_numpy=True, device=device)
    print(f"Model verified! Embedding dimension: {len(test_embedding)}")
    print(" all-MiniLM-L6-v2 ready")
except Exception as e:
    print(f" Error: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Try to download BioBERT (optional, won't fail if it fails)
print("\nDownloading BioBERT (optional)...")
try:
    SentenceTransformer("dmis-lab/biobert-base-cased-v1.2", cache_folder=cache_dir)
    print(" BioBERT model cached successfully")
except Exception as e:
    print(f"⚠️ BioBERT download failed (will use default model): {e}")
