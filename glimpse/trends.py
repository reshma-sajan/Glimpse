"""
glimpse/trends.py
Trend scoring — detect which visual clusters are growing or shrinking.
"""

import numpy as np
from collections import Counter
from typing import Optional


class TrendAnalyser:
    """
    Scores aesthetic clusters by their growth trajectory.

    Works in two modes:
    - Temporal: if items have timestamps, tracks cluster growth over time windows.
    - Distributional: compares cluster density near the query vs. the global baseline.
    """

    @staticmethod
    def score_by_recency(
        labels: np.ndarray,
        timestamps: np.ndarray,
        n_recent: int = 100,
    ) -> dict[int, float]:
        """
        Score clusters by how over/under-represented they are in recent items
        vs. the full dataset. A score > 1.0 means the cluster is trending up.
        """
        total_counts = Counter(labels[labels != -1])
        total_n = sum(total_counts.values())

        # Get the most recent items
        recent_indices = np.argsort(timestamps)[-n_recent:]
        recent_labels = labels[recent_indices]
        recent_counts = Counter(recent_labels[recent_labels != -1])
        recent_n = sum(recent_counts.values())

        if recent_n == 0 or total_n == 0:
            return {}

        scores = {}
        for cluster_id in total_counts:
            global_share = total_counts[cluster_id] / total_n
            recent_share = recent_counts.get(cluster_id, 0) / recent_n
            # Ratio of recent share to global share
            scores[cluster_id] = round(recent_share / global_share, 3) if global_share > 0 else 0.0

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def score_by_density(
        labels: np.ndarray,
        distances_to_query: np.ndarray,
        top_k: int = 50,
    ) -> dict[int, float]:
        """
        Score clusters by how concentrated they are among the nearest
        neighbours of a query image. Useful when there's no temporal data.
        """
        nearest_indices = np.argsort(distances_to_query)[:top_k]
        nearest_labels = labels[nearest_indices]

        total_counts = Counter(labels[labels != -1])
        total_n = sum(total_counts.values())

        nearest_counts = Counter(nearest_labels[nearest_labels != -1])
        nearest_n = sum(nearest_counts.values())

        if nearest_n == 0 or total_n == 0:
            return {}

        scores = {}
        for cluster_id in nearest_counts:
            global_share = total_counts.get(cluster_id, 0) / total_n
            local_share = nearest_counts[cluster_id] / nearest_n
            scores[cluster_id] = round(local_share / global_share, 3) if global_share > 0 else 0.0

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
