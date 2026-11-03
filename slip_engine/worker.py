import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from supabase import create_client, Client
from .config import EngineConfig
from .embedder import SlipEmbedder
from .ranker import RecommendationRanker
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger("SlipWorker")

class SlipWorker:
    def __init__(self):
        logger.info(f"Connecting to Supabase at {EngineConfig.SUPABASE_URL}...")
        self.supabase: Client = create_client(
            EngineConfig.SUPABASE_URL, EngineConfig.SUPABASE_KEY
        )
        self.embedder = SlipEmbedder()

    def process_unindexed_slips(self) -> int:
        """Finds slips lacking vector embeddings, computes dense vectors, and stores them."""
        # 1. Fetch all slips
        res = self.supabase.table("slips").select("id, topic, subtopic, trap_statement").execute()
        all_slips = res.data or []
        if not all_slips:
            return 0

        # 2. Fetch existing embeddings
        existing = self.supabase.table("slip_embeddings").select("slip_id").execute()
        indexed_ids = {item["slip_id"] for item in (existing.data or [])}

        # 3. Filter pending
        pending = [s for s in all_slips if s["id"] not in indexed_ids]
        if not pending:
            return 0

        logger.info(f"Found {len(pending)} unindexed slips. Generating CPU embeddings...")
        embeddings = self.embedder.encode_slips(pending)

        records = [
            {
                "slip_id": s["id"],
                "embedding": embeddings[i].tolist(),
                "model_version": "bge-base-cpu-v1",
            }
            for i, s in enumerate(pending)
        ]

        self.supabase.table("slip_embeddings").upsert(records).execute()
        logger.info(f"Indexed {len(records)} new slips in Supabase pgvector.")
        return len(records)

    def update_recommendation_feeds(self):
        """Recomputes personalized recommendation feeds for all active users."""
        users_res = self.supabase.table("profiles").select("id").execute()
        users = users_res.data or []
        if not users:
            return

        # Fetch all embeddings
        emb_res = self.supabase.table("slip_embeddings").select("slip_id, embedding").execute()
        embeddings_data = emb_res.data or []
        if not embeddings_data:
            return

        slip_ids = [e["slip_id"] for e in embeddings_data]
        matrix = np.array([e["embedding"] for e in embeddings_data], dtype=np.float32)

        total_updated = 0
        for u in users:
            uid = u["id"]
            # Fetch user interactions
            int_res = self.supabase.table("slip_interactions").select(
                "slip_id, interaction_type"
            ).eq("user_id", uid).execute()
            interactions = int_res.data or []

            # Compute ranked recommendations
            rankings = RecommendationRanker.compute_user_feed(
                user_id=uid,
                all_slip_ids=slip_ids,
                embedding_matrix=matrix,
                user_interactions=interactions,
                top_k=50,
            )

            if rankings:
                self.supabase.table("user_recommendation_feed").upsert(rankings).execute()
                total_updated += 1

        logger.info(f"Updated personalized recommendation feeds for {total_updated} users.")

    def run_12h_session(self):
        """Runs the continuous polling session for up to 11.5 hours on Kaggle CPU."""
        start_time = time.time()
        max_duration = EngineConfig.MAX_SESSION_HOURS * 3600
        interval = EngineConfig.CYCLE_INTERVAL_SECONDS

        logger.info(f"Starting 12-Hour Continuous Session at {datetime.now()}...")
        logger.info(f"Max duration: {EngineConfig.MAX_SESSION_HOURS} hrs. Polling interval: {interval}s.")

        cycle_count = 0
        while (time.time() - start_time) < max_duration:
            cycle_count += 1
            logger.info(f"--- Cycle #{cycle_count} ---")
            try:
                new_slips = self.process_unindexed_slips()
                self.update_recommendation_feeds()
            except Exception as e:
                logger.error(f"Error in cycle #{cycle_count}: {e}", exc_info=True)

            elapsed_hours = (time.time() - start_time) / 3600
            logger.info(f"Session elapsed: {elapsed_hours:.2f} / {EngineConfig.MAX_SESSION_HOURS} hrs.")
            logger.info(f"Sleeping for {interval}s...")
            time.sleep(interval)

        logger.info("12-Hour session completed gracefully. Handing off to next account.")

if __name__ == "__main__":
    worker = SlipWorker()
    worker.run_12h_session()
