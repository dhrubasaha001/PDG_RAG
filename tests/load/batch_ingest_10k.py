"""
tests/load/batch_ingest_10k.py
Load test — ingest 10K documents and report throughput.

Usage
-----
# 1. Start infrastructure
docker compose -f docker/docker-compose.yml up -d

# 2. Start API
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. Start Celery worker (separate terminal, Windows)
celery -A backend.workers.celery_app worker -Q high_priority,batch --pool=solo -l warning

# 4. Run load test
python tests/load/batch_ingest_10k.py --docs 10000 --batch 50 --api http://localhost:8000

Options
-------
--docs     Total documents to ingest (default: 10000)
--batch    Files per batch request (default: 50)
--api      API base URL (default: http://localhost:8000)
--real-dir Path to a directory of real PDF files (optional)
--seed     Seed file to replicate when generating synthetic docs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDG_RAG 10K load test")
    p.add_argument("--docs",     type=int,   default=10_000,              help="Total docs to ingest")
    p.add_argument("--batch",    type=int,   default=50,                  help="Files per batch request")
    p.add_argument("--api",      type=str,   default="http://localhost:8000", help="API base URL")
    p.add_argument("--real-dir", type=str,   default=None,                help="Directory of real files to use")
    p.add_argument("--seed",     type=str,   default=None,                help="Seed PDF/TXT to replicate")
    p.add_argument("--poll-interval", type=int, default=30,               help="Job poll interval in seconds")
    return p.parse_args()


def _generate_synthetic_docs(n: int, tmp_dir: Path, seed_path: Path | None) -> list[Path]:
    """
    Create n synthetic text files. If seed_path is provided, use its content
    as a base and vary each file by injecting a unique prefix.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    docs: list[Path] = []

    if seed_path and seed_path.exists() and seed_path.suffix.lower() == ".txt":
        base_content = seed_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Default seed: a few paragraphs that simulate an academic doc
        base_content = (
            "Abstract: This document describes algorithms and data structures. "
            "We analyse time complexity using the Master Theorem. "
            "Divide and conquer strategies reduce problems recursively. "
            "Dynamic programming stores overlapping subproblem results. "
            "Graph algorithms traverse edges to find shortest paths. "
            "Sorting algorithms achieve O(n log n) in the average case. "
        ) * 40   # ~300 tokens per doc

    for i in range(n):
        p = tmp_dir / f"synthetic_{i:06d}.txt"
        p.write_text(
            f"[Document {i}]\n\n{base_content}",
            encoding="utf-8",
        )
        docs.append(p)
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n} synthetic docs...")

    return docs


def _upload_batch(api_url: str, files: list[Path]) -> dict:
    """Upload one batch, return the response JSON."""
    import urllib.request, urllib.error
    boundary = "----LoadTestBoundary"
    parts = []
    for f in files:
        data = f.read_bytes()
        ext = f.suffix.lower()
        mime = "text/plain" if ext in {".txt", ".md"} else "application/pdf"
        parts.append(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="files"; filename="{f.name}"\r\n'
            f'Content-Type: {mime}\r\n\r\n'.encode()
            + data
            + b'\r\n'
        )
    body = b''.join(parts) + f'--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        f"{api_url}/ingest/batch",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def _poll_job(api_url: str, job_id: str) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{api_url}/jobs/{job_id}", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _memory_mb() -> float:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("PDG_RAG — 10K Document Ingestion Load Test")
    print("=" * 60)
    print(f"Target docs : {args.docs:,}")
    print(f"Batch size  : {args.batch}")
    print(f"API URL     : {args.api}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Check API is reachable
    # ------------------------------------------------------------------ #
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{args.api}/health", timeout=5) as r:
            health = json.loads(r.read())
        print(f"API health  : {health.get('status', 'unknown')}")
        deps = health.get("dependencies", {})
        print(f"Weaviate    : {deps.get('weaviate', '?')}")
        print(f"Redis       : {deps.get('redis', '?')}")
        if deps.get("weaviate") != "up" or deps.get("redis") != "up":
            print("\nWARNING: Some dependencies are down. Ingestion tasks will fail.")
    except Exception as e:
        print(f"ERROR: API not reachable at {args.api}: {e}")
        print("Start the API with: uvicorn backend.main:app --reload --port 8000")
        sys.exit(1)

    print()

    # ------------------------------------------------------------------ #
    # 2. Prepare documents
    # ------------------------------------------------------------------ #
    if args.real_dir:
        doc_dir = Path(args.real_dir)
        all_docs = sorted(doc_dir.rglob("*.pdf"))[:args.docs]
        if len(all_docs) < args.docs:
            print(f"Found {len(all_docs)} PDFs in {doc_dir} (requested {args.docs})")
        print(f"Using {len(all_docs):,} real documents from {doc_dir}")
    else:
        seed = Path(args.seed) if args.seed else None
        tmp_dir = REPO_ROOT / "data" / "_load_test_tmp"
        print(f"Generating {args.docs:,} synthetic text documents in {tmp_dir}...")
        t0 = time.perf_counter()
        all_docs = _generate_synthetic_docs(args.docs, tmp_dir, seed)
        print(f"Generated in {time.perf_counter() - t0:.1f}s\n")

    total_docs   = len(all_docs)
    batches      = [all_docs[i:i + args.batch] for i in range(0, total_docs, args.batch)]
    job_ids: list[str] = []

    # ------------------------------------------------------------------ #
    # 3. Submit all batches
    # ------------------------------------------------------------------ #
    print(f"Submitting {len(batches)} batch(es) of ≤{args.batch} docs each...")
    submit_start = time.perf_counter()
    submitted = 0
    errors = 0

    for idx, batch in enumerate(batches, 1):
        resp = _upload_batch(args.api, batch)
        if "error" in resp:
            errors += 1
            if errors <= 5:
                print(f"  Batch {idx} error: {resp['error']}")
        else:
            jid = resp.get("job_id")
            if jid:
                job_ids.append(jid)
            submitted += resp.get("total", 0)

        if idx % 20 == 0 or idx == len(batches):
            elapsed = time.perf_counter() - submit_start
            rate = submitted / elapsed if elapsed > 0 else 0
            print(f"  Submitted {idx}/{len(batches)} batches | "
                  f"{submitted:,} docs | {rate:.1f} docs/s | "
                  f"mem: {_memory_mb():.0f} MB")

    submit_elapsed = time.perf_counter() - submit_start
    print(f"\nAll batches submitted in {submit_elapsed:.1f}s")
    print(f"Jobs created: {len(job_ids)} | Submit errors: {errors}\n")

    if not job_ids:
        print("No jobs created — nothing to poll. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Poll until all jobs complete
    # ------------------------------------------------------------------ #
    print(f"Polling {len(job_ids)} jobs every {args.poll_interval}s...")
    process_start = time.perf_counter()
    poll_count = 0

    while True:
        time.sleep(args.poll_interval)
        poll_count += 1

        total_processed = 0
        total_failed    = 0
        total_docs_in_jobs = 0
        completed_jobs  = 0

        for jid in job_ids:
            j = _poll_job(args.api, jid)
            if "error" not in j:
                total_processed    += j.get("processed_documents", 0)
                total_failed       += j.get("failed_documents", 0)
                total_docs_in_jobs += j.get("total_documents", 0)
                if j.get("status") in ("completed", "partial", "failed"):
                    completed_jobs += 1

        elapsed = time.perf_counter() - process_start
        rate = total_processed / elapsed * 60 if elapsed > 0 else 0
        pct  = 100 * total_processed / max(total_docs_in_jobs, 1)

        print(
            f"  Poll {poll_count:3d} | "
            f"{total_processed:6,}/{total_docs_in_jobs:,} ({pct:.1f}%) indexed | "
            f"{total_failed:,} failed | "
            f"{rate:.1f} docs/min | "
            f"elapsed: {elapsed:.0f}s | "
            f"jobs done: {completed_jobs}/{len(job_ids)} | "
            f"mem: {_memory_mb():.0f} MB"
        )

        if completed_jobs == len(job_ids):
            print("\nAll jobs completed.")
            break

        # Safety exit after 2 hours
        if elapsed > 7200:
            print("\nTimeout: 2 hours elapsed — stopping poll.")
            break

    # ------------------------------------------------------------------ #
    # 5. Final report
    # ------------------------------------------------------------------ #
    total_elapsed = time.perf_counter() - submit_start
    throughput_docs_min = total_processed / (total_elapsed / 60) if total_elapsed > 0 else 0

    print()
    print("=" * 60)
    print("LOAD TEST REPORT")
    print("=" * 60)
    print(f"Total documents submitted : {submitted:,}")
    print(f"Total documents indexed   : {total_processed:,}")
    print(f"Total documents failed    : {total_failed:,}")
    print(f"Success rate              : {100*total_processed/max(submitted,1):.1f}%")
    print(f"Total wall time           : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Throughput                : {throughput_docs_min:.1f} docs/min")
    print(f"Estimated time for 1M docs: {1_000_000/max(throughput_docs_min,1)/60:.1f} hours")
    print(f"Peak memory (this process): {_memory_mb():.0f} MB")
    print("=" * 60)

    # Cleanup synthetic docs
    if not args.real_dir:
        import shutil
        tmp_dir = REPO_ROOT / "data" / "_load_test_tmp"
        if tmp_dir.exists():
            shutil.rmtree(str(tmp_dir))
            print("Synthetic docs cleaned up.")


if __name__ == "__main__":
    main()
