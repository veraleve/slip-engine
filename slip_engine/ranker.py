import logging
from typing import List, Dict, Any, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RecommendationRanker:
    """
    Computes personalized mistake recommendation feeds based on individual
    conceptual blindspots and 'Same' reaction histories.
    Enforces strict paper isolation (e.g. CS/DA users never see EE/EC slips).
    """

    @staticmethod
    def compute_user_feed(
        user_id: str,
        all_slip_ids: List[str],
        embedding_matrix: np.ndarray,
        user_interactions: List[Dict[str, Any]],
        allowed_exam_tags: Optional[Set[str]] = None,
        slip_exam_map: Optional[Dict[str, str]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity rankings for a single user with strict paper isolation.
        """
        if len(all_slip_ids) == 0 or embedding_matrix.shape[0] == 0:
            return []

        # 1. Filter by allowed papers if provided
        candidate_indices = []
        for i, sid in enumerate(all_slip_ids):
            if allowed_exam_tags and slip_exam_map:
                exam = slip_exam_map.get(sid, "")
                if exam and exam not in allowed_exam_tags:
                    continue
            candidate_indices.append(i)

        if not candidate_indices:
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
                raw_scores = np.dot(embedding_matrix, user_centroid.T).flatten()
            else:
                raw_scores = np.ones(len(all_slip_ids), dtype=np.float32) * 0.5
        else:
            raw_scores = np.ones(len(all_slip_ids), dtype=np.float32) * 0.5

        rankings = []
        for idx in candidate_indices:
            sid = all_slip_ids[idx]
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
