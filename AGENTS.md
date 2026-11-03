# AGENTS.md — Developer & AI Agent Guidelines for Slip Engine

## 1. Engine Mission
The **Slip Recommendation Engine** runs on Kaggle High-RAM CPU instances (0 GPU hours consumed) in continuous 11.5-hour sessions. It encodes newly submitted mistake blunders with canonical GATE 2027 syllabus hierarchy, computes cosine similarity with user vulnerability centroids, and pushes ranked feeds into Supabase `pgvector`.

---

## 2. Directory Structure

```
slip_engine/
├── config.py           # Supabase connection keys, batch sizes, CPU thread count
├── gate_syllabus.py    # Canonical IIT Madras 2027 taxonomy (CS, DA, EC, EE) & prompt formatter
├── embedder.py         # CPU BAAI/bge-base-en-v1.5 dense vector encoder (768-dim)
├── ranker.py           # Cosine similarity ranker with strict paper isolation
└── worker.py           # 12-hour continuous polling loop
render_api/
└── main.py             # FastAPI microservice for on-demand trigger & healthcheck
tests/
└── test_engine.py      # Unit tests for embedding generation and paper isolation
```

---

## 3. Strict Rules

### 3.1 Strict Paper Isolation
Recommendation feeds must enforce user paper boundaries:
- A user registered for `GATE CS 2027` and `GATE DA 2027` must **only** receive candidate recommendations where `slip.exam_tag in {"GATE CS 2027", "GATE DA 2027"}`.
- Cross-contamination with `GATE EC 2027` or `GATE EE 2027` is strictly forbidden.

### 3.2 Syllabus Context Prompting
Always format slip texts using `format_slip_embedding_prompt()`:
```python
"GATE Syllabus Context: [GATE DA 2027] Machine Learning > Support Vector Machine (SVM) | Blunder Trap: Forgot C parameter inversion. Conceptual Fix: High C penalizes slack heavily, resulting in narrow margin."
```

### 3.3 Kaggle CPU Resource Limits
- Never initialize GPU CUDA contexts.
- Force `torch.set_num_threads(4)` to saturate Kaggle CPU cores without thread thrashing.
- Keep memory utilization under 8 GB RAM.
