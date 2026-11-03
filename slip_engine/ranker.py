import logging
from typing import List, Dict, Any, Set
import numpy as np

logger = logging.getLogger(__name__)

class RecommendationRanker:
    """
    Computes personalized mistake recommendation feeds based on individual
    conceptual blindspots and 'Same' reaction histories.
    """

    @staticmethod
    def compute_user_feed(
        user_id: str,
        all_slip_ids: List[str],
        embedding_matrix: np.ndarray,
        user_interactions: List[Dict[str, Any]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity rankings for a single user.
        """
        if len(all_slip_ids) == 0 or embedding_matrix.shape[0] == 0:
            return []

        interacted_ids: Set[str] = {
            item["slip_id"] for item in user_interactions
        }
        # Traps the user fell for ('same') or saved ('keep') define their vulnerability centroid
        weak_slip_ids = [
            item["slip_id"]
            for item in user_interactions
            if item.get("interaction_type") in ("same", "keep")
        ]

        if weak_slip_ids:
            weak_indices = [
                i for i, sid in enumerate(all_slip_ids) if sid in weak_slip_ids
            ]
            if weak_indices:
                user_centroid = np.mean(
                    embedding_matrix[weak_indices], axis=0, keepdims=True
                )
                # Cosine similarity against all community slips
                raw_scores = np.dot(embedding_matrix, user_centroid.T).flatten()
            else:
                raw_scores = np.ones(len(all_slip_ids), dtype=np.float32) * 0.5
        else:
            # Cold start: neutral baseline
            raw_scores = np.ones(len(all_slip_ids), dtype=np.float32) * 0.5

        # Penalize already seen/interacted traps
        rankings = []
        for idx, sid in enumerate(all_slip_ids):
            score = float(raw_scores[idx])
            if sid in interacted_ids:
                score *= 0.15  # Heavily de-prioritize seen traps

            rankings.append({
                "user_id": user_id,
                "slip_id": sid,
                "rank_score": round(score, 4),
            })

        # Sort descending by rank score
        rankings.sort(key=lambda x: x["rank_score"], reverse=True)
        return rankings[:top_k]
