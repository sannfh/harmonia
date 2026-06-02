"""Tests for the generation service."""

import uuid
from unittest.mock import MagicMock

import pytest

from harmonia.services.generation import create_job, get_job


class TestJobLifecycle:
    def _make_db(self, job: object | None = None) -> MagicMock:
        db = MagicMock()
        db.get.return_value = job
        return db

    def test_create_job_adds_and_commits(self) -> None:
        db = MagicMock()
        returned_job = MagicMock()
        returned_job.id = uuid.uuid4()

        def fake_refresh(obj: object) -> None:
            pass

        db.refresh.side_effect = fake_refresh

        # Simulate db.add setting the id
        def fake_add(obj: object) -> None:
            obj.id = returned_job.id  # type: ignore[attr-defined]

        db.add.side_effect = fake_add
        db.refresh.side_effect = lambda obj: None

        from harmonia.infrastructure.models import GenerationJob

        real_job = GenerationJob(user_prompt="test prompt", status="pending")
        real_job.id = uuid.uuid4()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "harmonia.services.generation.GenerationJob",
                lambda **kw: real_job,
            )
            create_job(db, "test prompt")

        db.add.assert_called_once_with(real_job)
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(real_job)

    def test_get_job_returns_none_for_unknown_id(self) -> None:
        db = self._make_db(job=None)
        result = get_job(db, uuid.uuid4())
        assert result is None

    def test_get_job_calls_db_get(self) -> None:
        job_id = uuid.uuid4()
        mock_job = MagicMock()
        db = self._make_db(job=mock_job)
        result = get_job(db, job_id)
        assert result is mock_job
