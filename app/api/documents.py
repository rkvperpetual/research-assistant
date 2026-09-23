"""
app/api/documents.py — /documents endpoints.

POST /documents       — multipart file upload (one or many)
POST /documents/url   — ingest a web URL
GET  /documents       — list all ingested docs
GET  /documents/{job_id} — check ingestion job status
DELETE /documents/{doc_id} — remove a doc and its vectors
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi import status as http_status
from typing import List

from app.api.schemas import (
    BulkIngestResponse,
    DocumentSummary,
    IngestResponse,
    JobStatusResponse,
    UrlIngestRequest,
)
from app.ingestion.pipeline import (
    get_job,
    ingest_file,
    ingest_url,
    list_jobs,
    register_job,
)
from app.retrieval.vectorstore import delete_document, scroll_all_documents

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=BulkIngestResponse, status_code=http_status.HTTP_202_ACCEPTED)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Accept one or more files in any supported format.
    Ingestion runs in the background; returns job IDs immediately.
    """
    jobs: list[IngestResponse] = []

    for upload in files:
        job_id = str(uuid.uuid4())
        # Save to a temp file (BackgroundTask runs after response is sent)
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / (upload.filename or "upload")

        content = await upload.read()
        tmp_path.write_bytes(content)

        register_job(job_id, upload.filename or "upload")
        background_tasks.add_task(ingest_file, tmp_path, job_id)
        jobs.append(
            IngestResponse(job_id=job_id, status="processing", filename=upload.filename or "upload")
        )

    return BulkIngestResponse(jobs=jobs)


@router.post("/url", response_model=IngestResponse, status_code=http_status.HTTP_202_ACCEPTED)
async def ingest_document_url(
    payload: UrlIngestRequest,
    background_tasks: BackgroundTasks,
):
    """Fetch and ingest a web page by URL."""
    job_id = payload.job_id or str(uuid.uuid4())
    register_job(job_id, payload.url)
    background_tasks.add_task(ingest_url, payload.url, job_id)
    return IngestResponse(job_id=job_id, status="processing", filename=payload.url)


@router.get("", response_model=list[DocumentSummary])
async def list_documents():
    """List all ingested documents with chunk counts."""
    docs = scroll_all_documents()
    return [
        DocumentSummary(
            doc_id=d["doc_id"],
            filename=d["filename"],
            source_type=d["source_type"],
            chunk_count=d["chunk_count"],
            ingested_at=d.get("ingested_at", ""),
        )
        for d in docs
    ]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the ingestion job status."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job.get("filename", ""),
        chunks=job.get("chunks", 0),
        chunks_indexed=job.get("chunks_indexed", 0),
        progress=job.get("progress", 0),
        stage=job.get("stage", "queued"),
        doc_id=job.get("doc_id"),
        file_type=job.get("file_type"),
        error=job.get("error"),
    )


@router.delete("/{doc_id}", status_code=http_status.HTTP_200_OK)
async def delete_doc(doc_id: str):
    """Remove a document and all its vectors from Qdrant."""
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"deleted": deleted, "doc_id": doc_id}
