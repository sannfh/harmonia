"""GET /api/v1/jobs/{job_id} and GET /api/v1/files/{filename}."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from harmonia.api.dependencies import get_db, get_storage
from harmonia.api.schemas.job import JobResponse
from harmonia.infrastructure.storage import FileStorage
from harmonia.services.generation import get_job

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobResponse)
def poll_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    storage: FileStorage = Depends(get_storage),
) -> JobResponse:
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    midi_url = f"/api/v1/files/{job_id}.mid" if job.midi_path else None
    audio_url = f"/api/v1/files/{job_id}.mp3" if job.audio_path else None

    return JobResponse(
        id=job.id,
        created_at=job.created_at,
        updated_at=job.updated_at,
        status=job.status,
        user_prompt=job.user_prompt,
        music_params=job.music_params,
        midi_url=midi_url,
        audio_url=audio_url,
        error_message=job.error_message,
        duration_ms=job.duration_ms,
    )


@router.get("/files/{filename}")
def download_file(
    filename: str,
    storage: FileStorage = Depends(get_storage),
) -> FileResponse:
    path = storage._base / filename
    if not path.exists() or path.suffix not in {".mid", ".mp3"}:
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "audio/midi" if path.suffix == ".mid" else "audio/mpeg"
    return FileResponse(str(path), media_type=media_type, filename=filename)


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {"status": "ok", "db": db_status}
