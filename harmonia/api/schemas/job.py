"""Response schema for job status and file endpoints."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    status: str
    user_prompt: str
    music_params: dict | None = None
    midi_url: str | None = None
    audio_url: str | None = None
    error_message: str | None = None
    duration_ms: int | None = None

    model_config = {"from_attributes": True}
