"""MusicParameters: structured output from the DeepSeek interpreter."""

from pydantic import BaseModel, Field


class MusicParameters(BaseModel):
    key: str = Field(
        description="Root note of the key, e.g. 'C', 'F#', 'Bb'",
        examples=["A", "C", "F#"],
    )
    scale: str = Field(
        description="Scale/mode: 'major' or 'minor'",
        pattern="^(major|minor)$",
    )
    tempo_bpm: float = Field(
        description="Tempo in BPM",
        ge=40.0,
        le=240.0,
    )
    instruments: list[str] = Field(
        description="List of instrument names from the supported set",
        min_length=1,
        examples=[["Piano", "Bass"], ["Guitar", "Flute", "Bass"]],
    )

    @property
    def conditioning_prompt(self) -> str:
        """Build the conditioning string passed to the MIDI model."""
        from preprocessing.utils import build_conditioning_prompt

        return build_conditioning_prompt(
            self.key, self.scale, self.tempo_bpm, self.instruments
        )
