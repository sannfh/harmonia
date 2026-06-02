"""Tests for the API routes."""

import uuid
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from harmonia.api.dependencies import get_db, get_generator, get_settings, get_storage
from harmonia.api.main import app
from harmonia.api.settings import Settings


def _make_settings() -> Settings:
    return Settings(
        database_url="postgresql://x:x@localhost/x",
        deepseek_api_key="test-key",
        hf_model_repo="test/repo",
    )


def _make_job(status: str = "pending") -> MagicMock:
    job = MagicMock()
    job.id = uuid.uuid4()
    job.created_at = datetime.now(UTC)
    job.updated_at = datetime.now(UTC)
    job.user_prompt = "a sad piano piece"
    job.status = status
    job.music_params = None
    job.midi_path = None
    job.audio_path = None
    job.error_message = None
    job.duration_ms = None
    return job


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    mock_db = MagicMock()
    mock_generator = MagicMock()
    mock_storage = MagicMock()

    app.dependency_overrides[get_settings] = _make_settings
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_generator] = lambda: mock_generator
    app.dependency_overrides[get_storage] = lambda: mock_storage

    yield TestClient(app)

    app.dependency_overrides.clear()


class TestGenerateEndpoint:
    def test_returns_202_with_job_id(self, client: TestClient) -> None:
        mock_job = _make_job()

        with (
            patch("harmonia.api.routers.generate.create_job", return_value=mock_job),
            patch("harmonia.api.routers.generate.run_generation_pipeline"),
        ):
            resp = client.post("/api/v1/generate", json={"prompt": "a sad piano piece"})

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert "poll_url" in body

    def test_rejects_empty_prompt(self, client: TestClient) -> None:
        resp = client.post("/api/v1/generate", json={"prompt": "  "})
        assert resp.status_code == 422

    def test_rejects_missing_prompt(self, client: TestClient) -> None:
        resp = client.post("/api/v1/generate", json={})
        assert resp.status_code == 422


class TestJobsEndpoint:
    def test_returns_job_when_found(self, client: TestClient) -> None:
        job = _make_job("completed")

        with patch("harmonia.api.routers.jobs.get_job", return_value=job):
            resp = client.get(f"/api/v1/jobs/{job.id}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        with patch("harmonia.api.routers.jobs.get_job", return_value=None):
            resp = client.get(f"/api/v1/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
