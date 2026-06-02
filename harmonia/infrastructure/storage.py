"""File save/load helpers for MIDI and MP3 artifacts."""

import uuid
from pathlib import Path


class FileStorage:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    def midi_path(self, job_id: uuid.UUID) -> Path:
        return self._base / f"{job_id}.mid"

    def audio_path(self, job_id: uuid.UUID) -> Path:
        return self._base / f"{job_id}.mp3"

    def read_midi(self, job_id: uuid.UUID) -> bytes:
        return self.midi_path(job_id).read_bytes()

    def read_audio(self, job_id: uuid.UUID) -> bytes:
        return self.audio_path(job_id).read_bytes()
