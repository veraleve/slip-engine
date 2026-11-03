import pytest
import numpy as np
from slip_engine.embedder import SlipEmbedder
from slip_engine.ranker import RecommendationRanker
from slip_engine.gate_syllabus import format_slip_embedding_prompt, GATE_CS, GATE_DA, GATE_EE

def test_embedder_format_text():
    text = format_slip_embedding_prompt(
        exam_tag=GATE_CS,
        topic="Linear Algebra",
        subtopic="Eigenvalues & Eigenvectors",
        trap_statement="det(A)=0 means all lambda=0",
        fix_statement="Only at least one eigenvalue is 0"
    )
    assert "[GATE CS 2027]" in text
    assert "Linear Algebra > Eigenvalues & Eigenvectors" in text
    assert "Blunder Trap: det(A)=0 means all lambda=0" in text
    assert "Conceptual Fix: Only at least one eigenvalue is 0" in text

def test_recommendation_ranker_paper_isolation():
    slip_ids = ["s_cs_1", "s_da_1", "s_ee_1"]
    slip_exam_map = {
        "s_cs_1": GATE_CS,
        "s_da_1": GATE_DA,
        "s_ee_1": GATE_EE,
    }
    matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    interactions = [
        {"slip_id": "s_cs_1", "interaction_type": "same"}
    ]

    # User is preparing for CS + DA ONLY (EE must be excluded)
    rankings = RecommendationRanker.compute_user_feed(
        user_id="user_cs_da",
        all_slip_ids=slip_ids,
        embedding_matrix=matrix,
        user_interactions=interactions,
        allowed_exam_tags={GATE_CS, GATE_DA},
        slip_exam_map=slip_exam_map,
        top_k=5
    )

    result_ids = [r["slip_id"] for r in rankings]
    assert "s_cs_1" in result_ids
    assert "s_da_1" in result_ids
    assert "s_ee_1" not in result_ids, "EE slip leaked into CS/DA feed!"
