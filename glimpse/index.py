"""
glimpse/index.py
FAISS-based vector index for fast similarity search.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Optional


class ImageIndex:
    """Builds and queries a FAISS index over image embeddings."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim on L2-normed vectors)
        self.metadata: list[dict] = []  # Stores image paths, labels, etc.

    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        """Add embeddings and their metadata to the index."""
        assert embeddings.shape[0] == len(metadata), "Embeddings and metadata must match in length"
        assert embeddings.shape[1] == self.dim, f"Expected dim {self.dim}, got {embeddings.shape[1]}"

        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int = 10) -> list[dict]:
        """Find the k most similar images to the query embedding."""
        query = query.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results

    def save(self, path: str):
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """Load index and metadata from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    @property
    def size(self) -> int:
        return self.index.ntotal
