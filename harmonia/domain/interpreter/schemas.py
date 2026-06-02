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
        mood = "bright" if self.scale == "major" else "melancholic"
        if self.tempo_bpm < 70:
            feel = "slow"
        elif self.tempo_bpm < 120:
            feel = "moderate"
        else:
            feel = "upbeat"
        inst_str = ", ".join(self.instruments)
        bpm = round(self.tempo_bpm)
        return (
            f"Compose a {mood} {feel} piece in {self.key} {self.scale} at {bpm} BPM "
            f"featuring {inst_str}."
        )
