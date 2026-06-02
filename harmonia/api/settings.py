"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    deepseek_api_key: str
    hf_model_repo: str
    deepseek_base_url: str = "https://api.deepseek.com"
    storage_dir: Path = Path("storage")
    soundfont_path: Path = Path("soundfonts/GeneralUser_GS.sf2")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
