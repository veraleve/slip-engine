# Slip Engine — Recommendation & Embedding Pipeline

> **24/7 Continuous Kaggle CPU Worker & Microservice for Slip**

This repository powers the algorithmic discovery layer of [Slip](https://github.com/veraleve/slip-app). It pulls newly dropped exam blunders, generates dense semantic embeddings on Kaggle High-RAM CPU instances (consuming **0 GPU hours**), calculates individual conceptual blindspots, and continuously writes personalized recommendation feeds to Supabase PostgreSQL (`pgvector`).

---

## 1. Compute Architecture

* **Zero GPU Quota Overhead:** Runs exclusively on **Kaggle High-RAM CPU (4 vCPUs, 30 GB RAM)**. Kaggle provides unlimited CPU execution hours, allowing perpetual 24/7 batch recommendation processing.
* **Dual-Account 24/7 Relay:**
  * **Account 1:** Runs session from `00:00 UTC` to `11:30 UTC` (11.5 hours).
  * **Account 2:** Runs session from `12:00 UTC` to `23:30 UTC` (11.5 hours).
* **Render Microservice:** A FastAPI gateway that provides healthchecks, telemetry status, and on-demand trigger webhooks.
* **Supabase pgvector:** Stores 768-dimensional concept vectors with an **HNSW vector index** (`vector_cosine_ops`) for real-time nearest-neighbor retrieval.

---

## 2. Running in Kaggle (1-Click Execution)

1. Open a new Kaggle Notebook.
2. In the notebook settings:
   * **Accelerator:** None (CPU)
   * **Persistence:** Disabled
   * **Internet:** Enabled
3. Add your Supabase Service Key to Kaggle Secrets:
   * Click **Add-ons** > **Secrets** > Label: `SUPABASE_SERVICE_KEY`, Value: `<YOUR_SERVICE_ROLE_KEY>`.
4. Run the single execution command:
   ```bash
   !git clone https://github.com/veraleve/slip-engine.git
   %cd slip-engine
   !pip install -r requirements.txt
   !python -m slip_engine.worker
   ```
5. The notebook will process unindexed slips, recalculate user feeds, and run continuously for 11.5 hours before safely terminating.

---

## 3. Local Development & Testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/

# 3. Launch the Render FastAPI gateway locally
uvicorn render_api.main:app --reload --port 8000
```
