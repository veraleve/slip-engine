from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
from slip_engine.config import EngineConfig
from slip_engine.worker import SlipWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SlipAPI")

app = FastAPI(
    title="Slip Recommendation Microservice",
    version="1.0.0",
    description="Microservice API for triggering recommendation recomputations and telemetry sync."
)

worker = None

def get_worker():
    global worker
    if worker is None:
        worker = SlipWorker()
    return worker

@app.get("/")
def root():
    return {
        "service": "Slip Recommendation Engine",
        "status": "online",
        "supabase_endpoint": EngineConfig.SUPABASE_URL,
        "region": "ap-south-1"
    }

@app.get("/health")
def healthcheck():
    return {"status": "healthy"}

@app.post("/api/v1/trigger-cycle")
def trigger_batch_cycle(background_tasks: BackgroundTasks):
    """Triggers an out-of-band indexing and recommendation batch cycle."""
    w = get_worker()
    
    def run_cycle():
        try:
            logger.info("Executing on-demand batch cycle...")
            w.process_unindexed_slips()
            w.update_recommendation_feeds()
            logger.info("On-demand cycle complete.")
        except Exception as e:
            logger.error(f"Cycle execution failed: {e}")

    background_tasks.add_task(run_cycle)
    return {"message": "Batch cycle initiated in background"}
