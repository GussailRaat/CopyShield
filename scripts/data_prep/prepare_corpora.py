#!/usr/bin/env python3
"""
prepare_corpora.py
==================
Build the two text corpora used by CopyShield:

  data/books_corpus.txt          Protected corpus (D) -- 5 books used to induce
                                 memorization in the SFT baseline.
  data/neutral_books_corpus.txt  Neutral corpus       -- 5 books used for
                                 threshold calibration and as label-0 negatives
                                 for the activation classifier.

Pipeline (one script, three stages):

  1. download_book   -- fetch raw .txt from Project Gutenberg
  2. clean_text      -- strip Gutenberg headers/footers/artifacts
  3. build_corpus    -- concatenate cleaned books into a single corpus file

Usage:
    python scripts/data_prep/prepare_corpora.py
    python scripts/data_prep/prepare_corpora.py --only protected
    python scripts/data_prep/prepare_corpora.py --raw_dir data/raw_books \
                                                --clean_dir data/cleaned_books

Requires:
    requests
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Book registries (Project Gutenberg IDs)
# ---------------------------------------------------------------------------

PROTECTED_BOOKS = [
    ("pride_and_prejudice",      1342, "Pride and Prejudice"),
    ("frankenstein",               84, "Frankenstein"),
    ("dracula",                   345, "Dracula"),
    ("moby_dick",                2701, "Moby-Dick"),
    ("sherlock_holmes",          1661, "The Adventures of Sherlock Holmes"),
]

NEUTRAL_BOOKS = [
    ("alice_in_wonderland",        11, "Alice's Adventures in Wonderland"),
    ("crime_and_punishment",     2554, "Crime and Punishment"),
    ("the_great_gatsby",        64317, "The Great Gatsby"),
    ("romeo_and_juliet",         1513, "Romeo and Juliet"),
    ("wuthering_heights",         768, "Wuthering Heights"),
]


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def download_book(book_id: int, out_path: Path, force: bool = False,
                  retries: int = 3) -> Path:
    """Download a Project Gutenberg book by ID. Returns the path written.

    Project Gutenberg rate-limits rapid automated requests (transient 404s /
    connection resets), so each URL template is retried with exponential
    backoff before we give up.
    """
    if out_path.exists() and not force:
        print(f"  [cached] {out_path.name}")
        return out_path

    last_err = None
    for attempt in range(retries):
        for url_tmpl in GUTENBERG_URLS:
            url = url_tmpl.format(id=book_id)
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and len(r.text) > 10_000:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(r.text, encoding="utf-8")
                    print(f"  [ok] {out_path.name} ({len(r.text):,} chars) from {url}")
                    return out_path
                last_err = f"HTTP {r.status_code} / {len(r.text)} chars"
            except requests.RequestException as e:
                last_err = str(e)
            time.sleep(0.5)
        if attempt < retries - 1:
            backoff = 2.0 * (attempt + 1)
            print(f"  [retry] book {book_id} failed ({last_err}); waiting {backoff:.0f}s ...")
            time.sleep(backoff)

    raise RuntimeError(f"Failed to download book {book_id} after {retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

START_MARKERS = [
    re.compile(r"^\*\*\*\s*START OF .*?PROJECT GUTENBERG.*?\*\*\*\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*END\*THE SMALL PRINT.*$", re.IGNORECASE | re.MULTILINE),
]
END_MARKERS = [
    re.compile(r"^\*\*\*\s*END OF .*?PROJECT GUTENBERG.*?\*\*\*\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^End of (the )?Project Gutenberg.*$", re.IGNORECASE | re.MULTILINE),
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove header/footer added by Project Gutenberg."""
    start_idx = 0
    for pat in START_MARKERS:
        m = pat.search(text)
        if m:
            start_idx = max(start_idx, m.end())

    end_idx = len(text)
    for pat in END_MARKERS:
        m = pat.search(text, pos=start_idx)
        if m:
            end_idx = min(end_idx, m.start())

    return text[start_idx:end_idx]


def clean_text(raw: str) -> str:
    """Clean a raw Gutenberg book text:
       - strip header/footer
       - drop BOM and carriage returns
       - collapse 3+ consecutive blank lines to 2
       - strip leading/trailing whitespace
    """
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    text = strip_gutenberg_boilerplate(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# 3. Concatenate
# ---------------------------------------------------------------------------

def build_corpus(cleaned_paths: list[Path], out_path: Path) -> int:
    """Concatenate cleaned books into a single corpus file.
    Returns total character count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for p in cleaned_paths:
        parts.append(p.read_text(encoding="utf-8").strip())
    blob = "\n\n".join(parts) + "\n"
    out_path.write_text(blob, encoding="utf-8")
    return len(blob)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_books(
    books: list[tuple[str, int, str]],
    raw_dir: Path,
    clean_dir: Path,
    corpus_path: Path,
    force: bool,
) -> None:
    cleaned = []
    for slug, book_id, title in books:
        print(f"\n[{title}] (Gutenberg #{book_id})")
        raw_path = raw_dir / f"{slug}.txt"
        download_book(book_id, raw_path, force=force)

        clean_path = clean_dir / f"{slug}.txt"
        if clean_path.exists() and not force:
            print(f"  [cached clean] {clean_path.name}")
        else:
            cleaned_text = clean_text(raw_path.read_text(encoding="utf-8"))
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            clean_path.write_text(cleaned_text, encoding="utf-8")
            print(f"  [cleaned] {clean_path.name} ({len(cleaned_text):,} chars)")
        cleaned.append(clean_path)

    chars = build_corpus(cleaned, corpus_path)
    print(f"\n[corpus] {corpus_path} ({chars:,} chars)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",  default="data",                   help="Root data directory")
    p.add_argument("--raw_dir",   default=None,                     help="Where to save raw downloads (default: <data_dir>/raw_books)")
    p.add_argument("--clean_dir", default=None,                     help="Where to save cleaned books (default: <data_dir>/cleaned_books)")
    p.add_argument("--only",      choices=["protected", "neutral"], help="Build only one corpus")
    p.add_argument("--force",     action="store_true",              help="Re-download / re-clean even if files exist")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    targets = []

    if args.only in (None, "protected"):
        targets.append((
            "PROTECTED",
            PROTECTED_BOOKS,
            Path(args.raw_dir)   if args.raw_dir   else data_dir / "raw_books",
            Path(args.clean_dir) if args.clean_dir else data_dir / "cleaned_books",
            data_dir / "books_corpus.txt",
        ))
    if args.only in (None, "neutral"):
        targets.append((
            "NEUTRAL",
            NEUTRAL_BOOKS,
            Path(args.raw_dir)   if args.raw_dir   else data_dir / "raw_neutral_books",
            Path(args.clean_dir) if args.clean_dir else data_dir / "cleaned_neutral_books",
            data_dir / "neutral_books_corpus.txt",
        ))

    for label, books, raw, clean, corpus in targets:
        print("\n" + "=" * 70)
        print(f"{label} CORPUS")
        print("=" * 70)
        process_books(books, raw, clean, corpus, args.force)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
