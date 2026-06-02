"""Generation service: job lifecycle + pipeline orchestration."""

import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from harmonia.domain.audio.renderer import render_to_mp3
from harmonia.domain.interpreter.client import interpret
from harmonia.domain.interpreter.schemas import MusicParameters
from harmonia.domain.model.inference import HarmoniaGenerator
from harmonia.infrastructure.models import GenerationJob
from harmonia.infrastructure.storage import FileStorage


def create_job(db: Session, user_prompt: str) -> GenerationJob:
    job = GenerationJob(user_prompt=user_prompt, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: uuid.UUID) -> GenerationJob | None:
    return db.get(GenerationJob, job_id)


def run_generation_pipeline(
    job_id: uuid.UUID,
    user_prompt: str,
    db: Session,
    generator: HarmoniaGenerator,
    storage: FileStorage,
    deepseek_api_key: str,
    deepseek_base_url: str,
    soundfont_path: Path,
) -> None:
    """Execute the full generation pipeline and update the job row.

    Designed to run inside a FastAPI BackgroundTask.
    """
    started = time.monotonic()
    job = db.get(GenerationJob, job_id)
    if job is None:
        return

    _set_status(db, job, "running")

    try:
        params: MusicParameters = interpret(
            user_prompt, api_key=deepseek_api_key, base_url=deepseek_base_url
        )
        job.music_params = params.model_dump()
        db.commit()

        midi_path = storage.midi_path(job_id)
        generator.generate_midi(params.conditioning_prompt, midi_path)

        audio_path = storage.audio_path(job_id)
        render_to_mp3(midi_path, audio_path, soundfont=soundfont_path)

        job.midi_path = str(midi_path)
        job.audio_path = str(audio_path)
        job.duration_ms = int((time.monotonic() - started) * 1000)
        _set_status(db, job, "completed")

    except Exception as exc:
        job.error_message = str(exc)
        job.duration_ms = int((time.monotonic() - started) * 1000)
        _set_status(db, job, "failed")
        raise


def _set_status(db: Session, job: GenerationJob, status: str) -> None:
    job.status = status
    db.commit()
