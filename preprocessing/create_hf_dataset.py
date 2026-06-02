"""Pack the tokenized JSONL and push it to HuggingFace Hub as a dataset.

Usage:
    python preprocessing/create_hf_dataset.py \\
        --input data/train.jsonl \\
        --repo-id your-username/harmonia-midi \\
        --hf-token $HUGGINGFACE_TOKEN
"""

import argparse

from datasets import Dataset  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo, e.g. user/harmonia-midi",
    )
    parser.add_argument("--hf-token", required=True)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    dataset = Dataset.from_json(args.input)
    print(f"Loaded {len(dataset):,} examples")
    print(f"Sample:\n{dataset[0]['text'][:300]}\n...")

    print(f"Pushing to {args.repo_id}...")
    dataset.push_to_hub(args.repo_id, token=args.hf_token, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
