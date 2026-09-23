"""
scripts/seed_kb.py — One-shot script to ingest the sample knowledge base.

Usage:
    # From the project root (with .env loaded):
    python scripts/seed_kb.py

    # Or inside docker:
    docker compose exec app python scripts/seed_kb.py

Behaviour:
- Scans data/sample_docs/ for all files
- POSTs them to the running API at SEED_API_URL (default: http://localhost:8000)
- Polls until all jobs are done
- Idempotent: duplicate files are skipped by the API (SHA-256 dedup)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

API_URL = os.getenv("SEED_API_URL", "http://localhost:8000")
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_docs"
POLL_INTERVAL = 2
MAX_WAIT = 600  # 10 minutes

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".html", ".htm", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".tiff", ".csv", ".xlsx",
}


def main() -> None:
    if not SAMPLE_DIR.exists():
        print(f"[seed_kb] Sample docs directory not found: {SAMPLE_DIR}")
        print("Create data/sample_docs/ and add your documents there.")
        sys.exit(1)

    files = [
        f for f in SAMPLE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print("[seed_kb] No supported files found in", SAMPLE_DIR)
        sys.exit(0)

    print(f"[seed_kb] Found {len(files)} files to ingest")
    client = httpx.Client(base_url=API_URL, timeout=30)

    # POST all files at once
    form_files = [("files", (f.name, f.read_bytes(), "application/octet-stream")) for f in files]
    resp = client.post("/documents", files=form_files)
    resp.raise_for_status()
    jobs = resp.json()["jobs"]
    job_ids = {j["job_id"]: j["filename"] for j in jobs}
    print(f"[seed_kb] Submitted {len(job_ids)} ingestion jobs")

    # Poll until all done
    pending = set(job_ids)
    deadline = time.time() + MAX_WAIT
    while pending and time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        done = set()
        for job_id in list(pending):
            status_resp = client.get(f"/documents/{job_id}")
            data = status_resp.json()
            status = data.get("status")
            name = job_ids[job_id]
            if status == "done":
                print(f"  + {name} - {data.get('chunks', 0)} chunks")
                done.add(job_id)
            elif status in ("failed", "duplicate"):
                icon = "=" if status == "duplicate" else "x"
                print(f"  {icon} {name} - {status}: {data.get('error', '')}")
                done.add(job_id)
        pending -= done

    if pending:
        print(f"[seed_kb] Timed out; {len(pending)} jobs still pending")
    else:
        print("[seed_kb] All files ingested successfully")

    # Print health summary
    health = client.get("/health").json()
    print(f"[seed_kb] Health: {health}")


if __name__ == "__main__":
    main()
