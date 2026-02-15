import json
import sys
from typing import Dict, Iterable, Set

import requests

SPLITS = {
    "train": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train",
    "dev": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev",
    "test": "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test",
}


def fetch_lines(url: str) -> Iterable[str]:
    """Download and yield JSON strings line by line."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    for line in response.text.splitlines():
        if line.strip():
            yield line


def summarize_quality() -> Dict[str, Dict[str, int]]:
    summary = {}
    overall_unique_docs: Set[str] = set()
    total_samples = 0

    for split, url in SPLITS.items():
        split_samples = 0
        split_unique_docs: Set[str] = set()
        error_count = 0

        for raw_line in fetch_lines(url):
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                error_count += 1
                print(f"[{split}] JSONDecodeError: {exc}")
                continue

            split_samples += 1
            article = sample.get("article", "").strip()
            if article:
                split_unique_docs.add(article)
                overall_unique_docs.add(article)

        total_samples += split_samples
        summary[split] = {
            "samples": split_samples,
            "unique_docs": len(split_unique_docs),
            "json_errors": error_count,
        }

    summary["overall"] = {
        "samples": total_samples,
        "unique_docs": len(overall_unique_docs),
    }

    return summary


def main():
    summary = summarize_quality()
    print("\nQuALITY dataset statistics:")
    for split in ("train", "dev", "test"):
        data = summary.get(split, {})
        print(
            f"- {split:5s}: number of data samples={data.get('samples', 0):6d}, "
            f"unique documents={data.get('unique_docs', 0):6d}, "
            f"parse errors={data.get('json_errors', 0):4d}"
        )

    overall = summary["overall"]
    print(
        f"\nOverall: number of data samples={overall['samples']}, "
        f"unique_documents={overall['unique_docs']}"
    )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Failed to download data: {exc}", file=sys.stderr)
        sys.exit(1)

