# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ingest a directory of PDFs into the Chat Q&A Core document store."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import requests


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "SE"
DEFAULT_API_URL = "http://localhost:8102/v1/chatqna"


def find_pdfs(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"PDF directory does not exist: {source}")

    pdfs = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        raise ValueError(f"No PDFs found in {source}")

    if any(path.is_symlink() for path in pdfs):
        raise ValueError("PDF symlinks are not supported")

    duplicates = [name for name, count in Counter(path.name for path in pdfs).items() if count > 1]
    if duplicates:
        raise ValueError(f"PDF filenames must be unique across folders: {', '.join(duplicates)}")

    return pdfs


def ingest_pdfs(source: Path, api_url: str, session: requests.Session) -> int:
    pdfs = find_pdfs(source)
    endpoint = f"{api_url.rstrip('/')}/documents"

    response = session.get(endpoint, timeout=(5, 30))
    response.raise_for_status()
    existing = response.json()["metadata"]["documents"]
    if not isinstance(existing, list) or not all(isinstance(name, str) for name in existing):
        raise ValueError("Invalid document list returned by the service")

    indexed = skipped = failed = 0
    for pdf in pdfs:
        relative_path = pdf.relative_to(source)
        if pdf.name in existing:
            print(f"Already indexed: {relative_path}")
            skipped += 1
            continue

        try:
            with pdf.open("rb") as document:
                result = session.post(
                    endpoint,
                    files={"files": (pdf.name, document, "application/pdf")},
                    timeout=(5, 1800),
                )
            result.raise_for_status()
            documents = result.json()["metadata"]["documents"]
            if not isinstance(documents, list) or pdf.name not in documents:
                raise ValueError("Upload response did not confirm the document")
            existing.append(pdf.name)
            indexed += 1
            print(f"Indexed: {relative_path}")
        except (OSError, requests.RequestException, ValueError, KeyError, TypeError) as error:
            failed += 1
            print(f"Failed: {relative_path}: {error}", file=sys.stderr)

    print(f"PDFs: {indexed} indexed, {skipped} already indexed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat Q&A Core API base URL")
    args = parser.parse_args()

    try:
        with requests.Session() as session:
            return ingest_pdfs(args.source, args.api_url, session)
    except (OSError, requests.RequestException, ValueError, KeyError, TypeError) as error:
        print(f"Cannot load PDFs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())