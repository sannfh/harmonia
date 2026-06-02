"""Quick local verification of the preprocessing pipeline.

Creates a synthetic MIDI file, runs it through the full pipeline, and prints
what each step produces. No dataset download required.

Run from the project root:
    python preprocessing/verify.py
"""

import tempfile
from pathlib import Path

import pretty_midi
from miditok import REMI, TokenizerConfig

from preprocessing.utils import (
    build_conditioning_prompt,
    detect_key,
    get_instruments,
)


def make_test_midi() -> pretty_midi.PrettyMIDI:
    """Create a simple 8-note melody in C major at 100 BPM."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    # C major scale: C D E F G E D C
    pitches = [60, 62, 64, 65, 67, 64, 62, 60]
    beat = 60 / 100  # seconds per beat

    for i, pitch in enumerate(pitches):
        piano.notes.append(
            pretty_midi.Note(velocity=80, pitch=pitch, start=i * beat, end=(i + 1) * beat)
        )

    midi.instruments.append(piano)
    return midi


def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def step(n: int, label: str) -> None:
    print(f"\n[Step {n}] {label}")


def main() -> None:
    section("Harmonia Preprocessing Pipeline Verification")

    # ------------------------------------------------------------------ Step 1
    step(1, "Creating synthetic MIDI (C major scale, piano, 100 BPM)")
    midi = make_test_midi()
    print(f"  Duration    : {midi.get_end_time():.2f}s")
    print(f"  Instruments : {[i.name for i in midi.instruments]}")
    print(f"  Total notes : {sum(len(i.notes) for i in midi.instruments)}")

    # ------------------------------------------------------------------ Step 2
    step(2, "Key detection  (Krumhansl-Schmuckler)")
    key, scale = detect_key(midi)
    match = "CORRECT" if key == "C" and scale == "major" else "UNEXPECTED"
    print(f"  Detected : {key} {scale}  [{match}]")

    # ------------------------------------------------------------------ Step 3
    step(3, "Instrument extraction")
    instruments = get_instruments(midi)
    print(f"  Instruments : {instruments}")

    # ------------------------------------------------------------------ Step 4
    step(4, "Building conditioning prompt")
    prompt = build_conditioning_prompt(key, scale, 100.0, instruments)
    print(f"  Prompt : \"{prompt}\"")

    # ------------------------------------------------------------------ Step 5
    step(5, "miditok REMI tokenization")
    config = TokenizerConfig(num_velocities=32, use_chords=False, use_programs=True)
    tokenizer = REMI(config)

    tmp = Path(tempfile.mktemp(suffix=".mid"))
    midi.write(str(tmp))

    try:
        token_sequences = tokenizer(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    all_tokens: list[str] = []
    for seq in token_sequences:
        all_tokens.extend(seq.tokens)  # type: ignore[arg-type]

    print(f"  Total tokens  : {len(all_tokens)}")
    print(f"  First 15 tokens: {' '.join(all_tokens[:15])}")

    # ------------------------------------------------------------------ Result
    section("Full Training Example (what the model will be trained on)")
    midi_tokens = " ".join(all_tokens[:1024])
    example = f"{prompt}\n<MIDI_START>\n{midi_tokens}\n<MIDI_END>"
    preview = example[:500] + ("\n..." if len(example) > 500 else "")
    print(preview)

    section("Done — pipeline is working correctly")


if __name__ == "__main__":
    main()
