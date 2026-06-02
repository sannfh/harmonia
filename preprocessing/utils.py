"""Shared utilities for MIDI preprocessing: key detection and instrument mapping."""

import numpy as np
import pretty_midi

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Maps General MIDI program ranges to our supported instrument names
_PROGRAM_TO_INSTRUMENT: list[tuple[range, str]] = [
    (range(0, 8), "Piano"),
    (range(8, 16), "Bells"),
    (range(24, 32), "Guitar"),
    (range(32, 40), "Bass"),
    (range(40, 56), "Strings"),
    (range(56, 64), "Trumpet"),
    (range(64, 72), "Sax"),
    (range(72, 80), "Flute"),
    (range(80, 96), "Synth"),
]

SUPPORTED_INSTRUMENTS = {name for _, name in _PROGRAM_TO_INSTRUMENT} | {"Drums"}


def detect_key(midi: pretty_midi.PrettyMIDI) -> tuple[str, str]:
    """Return (note, scale) using the Krumhansl-Schmuckler algorithm.

    Correlates the pitch-class histogram of all notes against major and minor
    key profiles for all 12 roots. Falls back to C major on empty input.
    """
    pitch_counts = np.zeros(12)
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch_counts[note.pitch % 12] += note.end - note.start

    if pitch_counts.sum() == 0:
        return "C", "major"

    pitch_counts /= pitch_counts.sum()

    best_key, best_scale, best_corr = "C", "major", -np.inf
    for i in range(12):
        major_corr = float(np.corrcoef(pitch_counts, np.roll(_MAJOR, i))[0, 1])
        minor_corr = float(np.corrcoef(pitch_counts, np.roll(_MINOR, i))[0, 1])
        if major_corr > best_corr:
            best_corr, best_key, best_scale = major_corr, _NOTE_NAMES[i], "major"
        if minor_corr > best_corr:
            best_corr, best_key, best_scale = minor_corr, _NOTE_NAMES[i], "minor"

    return best_key, best_scale


def get_instruments(midi: pretty_midi.PrettyMIDI) -> list[str]:
    """Return sorted list of unique supported instrument names present in the MIDI."""
    names: set[str] = set()
    for instrument in midi.instruments:
        if instrument.is_drum:
            names.add("Drums")
            continue
        for prog_range, name in _PROGRAM_TO_INSTRUMENT:
            if instrument.program in prog_range:
                names.add(name)
                break
    return sorted(names)


def build_conditioning_prompt(
    key: str,
    scale: str,
    tempo_bpm: float,
    instruments: list[str],
) -> str:
    """Build a natural language conditioning prefix from extracted MIDI parameters."""
    mood = "bright" if scale == "major" else "melancholic"
    if tempo_bpm < 70:
        feel = "slow"
    elif tempo_bpm < 120:
        feel = "moderate"
    else:
        feel = "upbeat"

    inst_str = ", ".join(instruments) if instruments else "piano"
    bpm = round(tempo_bpm)
    return (
        f"Compose a {mood} {feel} piece in {key} {scale} at {bpm} BPM "
        f"featuring {inst_str}."
    )
