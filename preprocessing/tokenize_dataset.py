"""Tokenize filtered MIDI files with miditok REMI and build training examples.

Each example is a JSON object with a single "text" field:
    {conditioning prompt}
    <MIDI_START>
    {space-separated REMI token names}
    <MIDI_END>

Output is a JSONL file (one JSON object per line).

Re-running is safe: already-processed files are detected from the existing
output and skipped, so a disconnected session can resume from where it left off.

Usage:
    python -m preprocessing.tokenize_dataset \\
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
        return json.dumps({"text": text, "source": midi_path_str})

    except Exception:
        return None


def _load_done_paths(output: Path) -> set[str]:
    """Return the set of source paths already written to the output file."""
    if not output.exists():
        return set()
    done: set[str] = set()
    with open(output) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "source" in obj:
                    done.add(obj["source"])
            except json.JSONDecodeError:
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", type=Path, default=Path("data/filtered.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    midi_paths = args.file_list.read_text().splitlines()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_paths(args.output)
    if done:
        print(f"Resuming: {len(done):,} files already done, skipping them.")
    remaining = [p for p in midi_paths if p not in done]
    print(f"Tokenizing {len(remaining):,} files with {args.workers} workers...")

    if not remaining:
        print("Nothing to do.")
        return

    process_fn = partial(_process_file, max_tokens=args.max_tokens)
    success = 0
    with mp.Pool(args.workers) as pool, open(args.output, "a") as out:
        for result in tqdm(
            pool.imap(process_fn, remaining, chunksize=32),
            total=len(remaining),
            desc="Tokenizing",
            unit="files",
        ):
            if result is not None:
                out.write(result + "\n")
                success += 1

    total_done = len(done) + success
    print(f"Done: {success:,} new examples written ({total_done:,} total) → {args.output}")


if __name__ == "__main__":
    main()
