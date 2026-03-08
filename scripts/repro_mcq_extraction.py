#!/usr/bin/env python3
"""Reproduce MCQ extraction behavior for GPQA/OpenBench scorers.

Usage:
  uv run python scripts/repro_mcq_extraction.py
  uv run python scripts/repro_mcq_extraction.py --text-file /path/to/completion.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openbench.scorers.mcq import MCQ_PATTERNS, extract_mcq_answer


DEFAULT_TEXT = """The calculated answer is 11.

Answer seems to be A.

Answer: A
"""


def _load_text(text_file: str | None) -> str:
    if not text_file:
        return DEFAULT_TEXT
    return Path(text_file).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to a file containing raw model completion text.",
    )
    args = parser.parse_args()

    text = _load_text(args.text_file)

    print("=== Input Text ===")
    print(text)
    print("=== Pattern Matches ===")
    found = False
    for idx, pattern in enumerate(MCQ_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        found = True
        print(
            f"[{idx}] match={match.group(0)!r} capture={match.group(1)!r} upper={match.group(1).upper()!r}"
        )

    if not found:
        print("(No English MCQ pattern matched)")

    extracted = extract_mcq_answer(text)
    print("=== Final Extracted Answer ===")
    print(repr(extracted))


if __name__ == "__main__":
    main()
