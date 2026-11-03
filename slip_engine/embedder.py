import logging
from typing import List, Dict, Any
import numpy as np
from .config import EngineConfig
from .gate_syllabus import format_slip_embedding_prompt

logger = logging.getLogger(__name__)

class SlipEmbedder:
    """
    CPU-Optimized Dense Embedding Engine for Slip Traps.
    Runs on Kaggle High-RAM CPU instances without consuming GPU quotas.
    """

    def __init__(self, model_name: str = EngineConfig.MODEL_NAME):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cpu"
        # Optimize CPU threads for Kaggle 4-core instances
        torch.set_num_threads(4)
        logger.info(f"Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Model loaded successfully.")

    def format_slip_text(
        self,
        exam_tag: str,
        topic: str,
        subtopic: str,
        trap_statement: str,
        fix_statement: str = ""
    ) -> str:
        """Formats metadata and blunder description into an optimal syllabus-aware embedding prompt."""
        return format_slip_embedding_prompt(
            exam_tag=exam_tag,
            topic=topic,
            subtopic=subtopic,
            trap_statement=trap_statement,
            fix_statement=fix_statement
        )

    def encode_slips(self, slips: List[Dict[str, Any]]) -> np.ndarray:
        """Encodes a list of slip dictionaries into normalized 768-dim numpy embeddings."""
        if not slips:
            return np.empty((0, EngineConfig.EMBEDDING_DIM))

        texts = [
            self.format_slip_text(
                exam_tag=s.get("exam_tag", "GATE"),
                topic=s.get("topic", "General"),
                subtopic=s.get("subtopic", ""),
                trap_statement=s.get("trap_statement", ""),
                fix_statement=s.get("fix_statement", "")
            )
            for s in slips
        ]

        logger.info(f"Encoding {len(texts)} slips on CPU with syllabus context...")
        embeddings = self.model.encode(
            texts,
            batch_size=EngineConfig.BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)
