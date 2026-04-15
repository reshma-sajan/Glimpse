"""
glimpse/cluster.py
UMAP + HDBSCAN clustering to discover aesthetic groupings.
"""

import numpy as np
import umap
import hdbscan
from typing import Optional


class AestheticClusterer:
    """Reduces CLIP embeddings to 2D and finds natural visual clusters."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 15,
        min_samples: int = 5,
    ):
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        self.embeddings_2d: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> dict:
        """Run UMAP reduction + HDBSCAN clustering. Returns cluster summary."""
        # Reduce to 2D
        self.embeddings_2d = self.reducer.fit_transform(embeddings)

        # Cluster in 2D space
        self.labels = self.clusterer.fit_predict(self.embeddings_2d)

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()

        return {
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "n_total": len(embeddings),
            "cluster_sizes": {
                int(label): int((self.labels == label).sum())
                for label in sorted(set(self.labels))
                if label != -1
            },
        }

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Return indices of items in a given cluster."""
        return np.where(self.labels == cluster_id)[0]

    def get_coordinates(self) -> np.ndarray:
        """Return the 2D UMAP coordinates for plotting."""
        if self.embeddings_2d is None:
            raise ValueError("Call fit() first")
        return self.embeddings_2d
