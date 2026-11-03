import pytest
import numpy as np
from slip_engine.embedder import SlipEmbedder
from slip_engine.ranker import RecommendationRanker

def test_embedder_format_text():
    embedder = SlipEmbedder.__new__(SlipEmbedder)
    text = embedder.format_slip_text(
        topic="Linear Algebra",
        subtopic="Eigenvalues",
        trap_statement="det(A)=0 means all lambda=0"
    )
    assert "Topic: Linear Algebra - Eigenvalues." in text
    assert "Trap: det(A)=0 means all lambda=0" in text

def test_recommendation_ranker():
    slip_ids = ["s1", "s2", "s3"]
    # 3 slips with 4-dim toy embeddings
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    # User fell for s1
    interactions = [
        {"slip_id": "s1", "interaction_type": "same"}
    ]

    rankings = RecommendationRanker.compute_user_feed(
        user_id="user_1",
        all_slip_ids=slip_ids,
        embedding_matrix=matrix,
        user_interactions=interactions,
        top_k=3
    )

    assert len(rankings) == 3
    # s1 was seen, so its score is penalized by 0.15
    s1_rank = next(r for r in rankings if r["slip_id"] == "s1")
    assert s1_rank["rank_score"] <= 0.2
