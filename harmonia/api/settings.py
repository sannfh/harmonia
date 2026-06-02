"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://harmonia:harmonia@localhost:5432/harmonia"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    hf_model_repo: str = ""
    storage_dir: Path = Path("storage")
    soundfont_path: Path = Path("soundfonts/GeneralUser_GS.sf2")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
