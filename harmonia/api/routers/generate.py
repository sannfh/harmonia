"""POST /api/v1/generate — submit a generation request."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from harmonia.api.dependencies import (
    get_db,
    get_generator,
    get_settings,
    get_storage,
)
from harmonia.api.schemas.generate import GenerateRequest
from harmonia.api.settings import Settings
from harmonia.domain.model.inference import HarmoniaGenerator
from harmonia.infrastructure.storage import FileStorage
from harmonia.services.generation import create_job, run_generation_pipeline

router = APIRouter()


@router.post("/generate", status_code=202)
def submit_generation(
    body: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    generator: HarmoniaGenerator = Depends(get_generator),
    storage: FileStorage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    job = create_job(db, body.prompt)
    job_id: uuid.UUID = job.id

    background_tasks.add_task(
        run_generation_pipeline,
        job_id=job_id,
        user_prompt=body.prompt,
        db=db,
        generator=generator,
        storage=storage,
        deepseek_api_key=settings.deepseek_api_key,
        deepseek_base_url=settings.deepseek_base_url,
        soundfont_path=settings.soundfont_path,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": str(job_id),
            "poll_url": f"/api/v1/jobs/{job_id}",
        },
    )
