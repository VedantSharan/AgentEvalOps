# Data Collection, Provenance & Licensing Note

**Project:** AgentEvalOps (Project 1)  
**Course:** CS 203 Software Tools and Techniques in AI  
**Repository:** `AgentEvalOps`  
**Date:** October 2026  

---

## 1. Consent and Licensing Register

### Data Provenance
All evaluation tasks in this repository (`data/raw/tasks.jsonl` and `data/processed/tasks_revised.jsonl`) are synthetically generated or curated benchmark tasks specifically designed to test tool-using LLM agent reasoning.
- **No Personal Identifiable Information (PII):** All names (e.g. "Rohan Deshmukh", "Aarav Patel"), employee IDs, and company records are fictitious and generated using synthetic distributions.
- **No Private or Proprietary Documents:** All virtual filesystem fixtures (e.g. `consulting_ledger.csv`, `server_metrics.csv`, `logs/cluster_events.json`) are self-contained, synthetic sandbox files.
- **API Terms of Service & Model Usage:**
  - Inference was performed via official free-tier developer APIs (Google Gemini Developer API, Groq Cloud API, NVIDIA NIM Catalog, Cohere Compatibility API, OpenRouter Free Tier, Cloudflare Workers AI).
  - All calls complied strictly with the respective providers' terms of service, rate limits, and non-commercial educational use guidelines. No scraping, account cycling, or terms violations were conducted.

---

## 2. End-to-End Data Flow

The diagram below traces how evaluation tasks and genuine model trajectories move through the system into `metadata.csv` and DVC storage:

```mermaid
flowchart TD
    subgraph Task_Generation ["Task Generation Layer (Part A)"]
        Seed["6 Benchmark Families<br/>(Multi-hop, Calc, Table, Chain, Audit, Ambiguous)"] --> Generator["code/tasks/revised_generator.py"]
        Generator --> Tasks["data/processed/tasks_revised.jsonl<br/>(30 Typed Tasks with Gold Answers)"]
    end

    subgraph Execution_Pipeline ["Agent Execution & Work Queue"]
        Tasks --> WorkQueue["SQLite Work Queue<br/>(Durable States & Fair Round-Robin)"]
        WorkQueue --> Worker["code/experiments/trajectory_worker.py"]
        Worker --> Providers["Verified LLM Providers<br/>(Gemini, Groq, NVIDIA, Cohere, OpenRouter)"]
        Providers <--> Tools["Typed Tool Layer<br/>(retrieve, calculate, query_records, read_file)"]
        Tools --> Traces["MLflow Traces & Trajectory Builder"]
    end

    subgraph Persistence_Validation ["Persistence & Manifest Layer"]
        Traces --> RawTraj["data/experiments/multi_provider_run/trajectories.jsonl<br/>(154 Real Agent Trajectories)"]
        RawTraj --> MLflowAdapter["code/tracing/export_trajectories_to_mlflow.py"]
        MLflowAdapter --> MLflowDB["mlruns/mlflow.db<br/>(174 Traces & 633 Spans for UI)"]
        Tasks & RawTraj --> MetaGen["code/experiments/generate_metadata.py"]
        MetaGen --> MetaCSV["data/metadata.csv<br/>(Traceability: source, time, tool, hash)"]
        Tasks & RawTraj --> Validator["code/validation/dataset_validator.py<br/>(12 Automated Integrity Checks)"]
        MetaCSV & Tasks & RawTraj --> DVC["DVC Data Versioning<br/>(Remote Tracking & Reproducibility)"]
    end
```

### Tracing Telemetry Architecture: Runner vs. Trajectory Worker

The telemetry system incorporates two complementary pipelines:
- **Synchronous Inline Tracing (`code/experiments/runner.py`):** For benchmark evaluations, the agent loop directly triggers `MLflowTracer` (`code/tracing/mlflow_tracer.py`), creating hierarchical spans (Root, LLM turn, Tool, Grader) live in MLflow.
- **Decoupled Batch Execution (`code/experiments/trajectory_worker.py`):** For high-throughput multi-provider runs across 10 remote APIs, the worker writes directly to append-only JSONL logs and an SQLite work queue. Decoupling live execution from synchronous SQLite logging prevents `sqlite3.OperationalError: database is locked` contention across concurrent worker processes and isolates network backoffs/retries.
- **Offline Trace Ingestion (`code/tracing/export_trajectories_to_mlflow.py`):** The ingestion adapter deterministically converts the recorded JSONL telemetry into MLflow traces in `mlruns/mlflow.db` with complete span hierarchies, token counts, and latencies.

---

## 3. Metadata Schema Specification (`data/metadata.csv`)

| Column | Description | Example |
|---|---|---|
| `sample_id` | Unique ID of task or trajectory run | `T1_multihop_01` or `93904b01-96f3-48e6...` |
| `sample_type` | Record type (`task` or `trajectory`) | `task` |
| `source` | Provenance generator or model identifier | `groq:openai/gpt-oss-120b` |
| `timestamp` | ISO 8601 creation/execution timestamp | `2026-10-21T11:00:00Z` |
| `collector` | Module or worker identity responsible | `code.experiments.trajectory_worker` |
| `tool_version` | Pinned software/tool release version | `v1.2.0` |
| `content_hash` | SHA-256 cryptographic digest | `e4f22a554f02da2780f08d2ba0942386...` |
| `extra_metadata` | Structured JSON containing operational attributes | `{"outcome": "error", "steps": 5}` |
