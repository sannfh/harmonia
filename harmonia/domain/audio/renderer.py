"""FluidSynth MIDI → MP3 renderer."""

import subprocess
import tempfile
from pathlib import Path

_DEFAULT_SOUNDFONT = Path("soundfonts/GeneralUser_GS.sf2")


def render_to_mp3(
    midi_path: Path,
    output_path: Path,
    soundfont: Path = _DEFAULT_SOUNDFONT,
    sample_rate: int = 44100,
) -> Path:
    """Render a MIDI file to MP3 using FluidSynth + pydub.

    FluidSynth must be installed (apt-get install fluidsynth on Linux).
    pydub must be installed and ffmpeg available for the WAV→MP3 step.

    Args:
        midi_path: Source MIDI file.
        output_path: Destination MP3 path.
        soundfont: Path to the .sf2 soundfont file.
        sample_rate: Audio sample rate in Hz.

    Returns:
        The path to the written MP3 file.
    """
    if not soundfont.exists():
        raise FileNotFoundError(
            f"Soundfont not found: {soundfont}. "
            "Download GeneralUser GS and place it at soundfonts/GeneralUser_GS.sf2"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        _fluidsynth_render(midi_path, wav_path, soundfont, sample_rate)
        _wav_to_mp3(wav_path, output_path)
    finally:
        wav_path.unlink(missing_ok=True)

    return output_path


def _fluidsynth_render(
    midi_path: Path,
    wav_path: Path,
    soundfont: Path,
    sample_rate: int,
) -> None:
    cmd = [
        "fluidsynth",
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(sample_rate),
        str(soundfont),
        str(midi_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FluidSynth failed (exit {result.returncode}):\n{result.stderr}"
        )


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    from pydub import AudioSegment  # type: ignore[import-untyped]

    audio = AudioSegment.from_wav(str(wav_path))
    audio.export(str(mp3_path), format="mp3", bitrate="192k")
