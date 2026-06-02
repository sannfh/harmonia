"""FastAPI dependency providers."""

from collections.abc import Generator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from harmonia.api.settings import Settings
from harmonia.domain.model.inference import HarmoniaGenerator
from harmonia.infrastructure.database import make_session_factory
from harmonia.infrastructure.storage import FileStorage


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  — values come from env/.env file


@lru_cache
def get_session_factory(
    settings: Settings = Depends(get_settings),
) -> sessionmaker[Session]:
    return make_session_factory(settings.database_url)


def get_db(
    factory: sessionmaker[Session] = Depends(get_session_factory),
) -> Generator[Session, None, None]:
    db = factory()
    try:
        yield db
    finally:
        db.close()


@lru_cache
def get_generator(settings: Settings = Depends(get_settings)) -> HarmoniaGenerator:
    return HarmoniaGenerator(settings.hf_model_repo)


@lru_cache
def get_storage(settings: Settings = Depends(get_settings)) -> FileStorage:
    return FileStorage(settings.storage_dir)
