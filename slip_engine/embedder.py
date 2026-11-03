import logging
from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .config import EngineConfig

logger = logging.getLogger(__name__)

class SlipEmbedder:
    """
    CPU-Optimized Dense Embedding Engine for Slip Traps.
    Runs on Kaggle High-RAM CPU instances without consuming GPU quotas.
    """

    def __init__(self, model_name: str = EngineConfig.MODEL_NAME):
        self.device = "cpu"
        # Optimize CPU threads for Kaggle 4-core instances
        torch.set_num_threads(4)
        logger.info(f"Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Model loaded successfully.")

    def format_slip_text(self, topic: str, subtopic: str, trap_statement: str) -> str:
        """Formats metadata and blunder description into an optimal embedding prompt."""
        sub = f" - {subtopic}" if subtopic else ""
        return f"Topic: {topic}{sub}. Trap: {trap_statement}"

    def encode_slips(self, slips: List[Dict[str, Any]]) -> np.ndarray:
        """Encodes a list of slip dictionaries into normalized 768-dim numpy embeddings."""
        if not slips:
            return np.empty((0, EngineConfig.EMBEDDING_DIM))

        texts = [
            self.format_slip_text(
                s.get("topic", "General"),
                s.get("subtopic", ""),
                s.get("trap_statement", "")
            )
            for s in slips
        ]

        logger.info(f"Encoding {len(texts)} slips on CPU...")
        embeddings = self.model.encode(
            texts,
            batch_size=EngineConfig.BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)
