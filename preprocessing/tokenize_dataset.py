"""Tokenize filtered MIDI files with miditok REMI and build training examples.

Each example is a JSON object with a single "text" field:
    {conditioning prompt}
    <MIDI_START>
    {space-separated REMI token names}
    <MIDI_END>

Output is a JSONL file (one JSON object per line).

Usage:
    python preprocessing/tokenize_dataset.py \\
        --file-list data/filtered.txt \\
        --output data/train.jsonl \\
        --max-tokens 1024
"""

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import pretty_midi
from miditok import REMI, TokenizerConfig
from tqdm import tqdm

from .utils import build_conditioning_prompt, detect_key, get_instruments

_TOKENIZER: REMI | None = None


def _get_tokenizer() -> REMI:
    global _TOKENIZER
    if _TOKENIZER is None:
        config = TokenizerConfig(
            num_velocities=32,
            use_chords=False,
            use_programs=True,
            use_time_signatures=False,
        )
        _TOKENIZER = REMI(config)
    return _TOKENIZER


def _process_file(midi_path_str: str, max_tokens: int = 1024) -> str | None:
    """Return a formatted training example string, or None on failure."""
    midi_path = Path(midi_path_str)
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        key, scale = detect_key(midi)
        instruments = get_instruments(midi)
        tempos = midi.get_tempo_changes()[1]
        tempo_bpm = float(tempos[0]) if len(tempos) > 0 else 120.0

        prompt = build_conditioning_prompt(key, scale, tempo_bpm, instruments)

        tokenizer = _get_tokenizer()
        token_sequences = tokenizer(midi_path)
        if not token_sequences:
            return None

        all_tokens: list[str] = []
        for seq in token_sequences:
            all_tokens.extend(seq.tokens)  # type: ignore[arg-type]

        if len(all_tokens) < 16:
            return None

        midi_tokens = " ".join(all_tokens[:max_tokens])
        text = f"{prompt}\n<MIDI_START>\n{midi_tokens}\n<MIDI_END>"
        return json.dumps({"text": text})

    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", type=Path, default=Path("data/filtered.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    midi_paths = args.file_list.read_text().splitlines()
    print(f"Tokenizing {len(midi_paths):,} files with {args.workers} workers...")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    process_fn = partial(_process_file, max_tokens=args.max_tokens)
    success = 0
    with mp.Pool(args.workers) as pool, open(args.output, "w") as out:
        for result in tqdm(
            pool.imap(process_fn, midi_paths, chunksize=32),
            total=len(midi_paths),
            desc="Tokenizing",
            unit="files",
        ):
            if result is not None:
                out.write(result + "\n")
                success += 1

    print(f"Done: {success:,} training examples → {args.output}")


if __name__ == "__main__":
    main()
