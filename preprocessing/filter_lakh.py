"""Download and filter the Lakh MIDI Dataset.

Downloads lmd_full.tar.gz, extracts it, and runs quality filters in parallel.
Writes the paths of passing files to a text file (one path per line).

Usage:
    python preprocessing/filter_lakh.py --data-dir data/lakh --output data/filtered.txt

If you already downloaded the archive elsewhere (e.g. Google Drive), pass --archive
to skip the download and extract from that path instead:

    python preprocessing/filter_lakh.py \\
        --data-dir  /content/lakh \\
        --archive   /content/drive/MyDrive/harmonia/data/lakh/lmd_full.tar.gz \\
        --output    /content/drive/MyDrive/harmonia/data/filtered.txt
"""

import argparse
import multiprocessing as mp
import tarfile
import urllib.request
from pathlib import Path

import pretty_midi

LMD_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"

MIN_DURATION_S = 20.0
MAX_DURATION_S = 600.0
MIN_NOTES = 50
MIN_MELODIC_TRACKS = 1


def _passes_filters(midi_path: Path) -> Path | None:
    """Return the path if the file passes all quality filters, else None."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None

    duration = midi.get_end_time()
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        return None

    melodic_tracks = [i for i in midi.instruments if not i.is_drum]
    if len(melodic_tracks) < MIN_MELODIC_TRACKS:
        return None

    total_notes = sum(len(i.notes) for i in melodic_tracks)
    if total_notes < MIN_NOTES:
        return None

    all_pitches = [n.pitch for i in melodic_tracks for n in i.notes]
    in_range = sum(1 for p in all_pitches if 36 <= p <= 96)
    if in_range / len(all_pitches) < 0.5:
        return None

    return midi_path


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:.1f}%  ({downloaded // 1_000_000} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _progress)
    print()


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} → {dest}  (this takes a few minutes...)")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    print("Extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lakh"),
        help="Directory to extract into and search for MIDI files",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to an already-downloaded lmd_full.tar.gz (skips download)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/filtered.txt"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    # Resolve archive path
    archive = args.archive if args.archive else args.data_dir / "lmd_full.tar.gz"
    extracted = args.data_dir / "lmd_full"

    if not args.skip_download and not archive.exists():
        _download(LMD_URL, archive)

    if not extracted.exists():
        _extract(archive, args.data_dir)
    else:
        print(f"Already extracted at {extracted}, skipping.")

    midi_files = list(extracted.rglob("*.mid")) + list(extracted.rglob("*.midi"))
    print(
        f"Found {len(midi_files):,} MIDI files, filtering with {args.workers} workers..."
    )

    with mp.Pool(args.workers) as pool:
        results = pool.map(_passes_filters, midi_files, chunksize=64)

    passing = [str(p) for p in results if p is not None]
    print(f"Passed: {len(passing):,} / {len(midi_files):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(passing))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
