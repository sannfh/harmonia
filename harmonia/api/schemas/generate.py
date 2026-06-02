"""Request schema for the generate endpoint."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=3,
        max_length=500,
        examples=["A melancholic jazz piece with piano and upright bass"],
    )
