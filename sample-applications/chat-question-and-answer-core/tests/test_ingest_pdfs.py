# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import requests

from scripts.ingest_pdfs import find_pdfs, ingest_pdfs


def response_with_documents(documents: list[str]) -> Mock:
    response = Mock()
    response.json.return_value = {"metadata": {"documents": documents}}
    return response


def test_find_pdfs_in_nested_directories(tmp_path):
    datasheets = tmp_path / "Datasheets"
    manuals = tmp_path / "Manuals"
    datasheets.mkdir()
    manuals.mkdir()
    datasheet = datasheets / "All_Data Sheet.pdf"
    manual = manuals / "COBOT Guide.PDF"
    datasheet.write_bytes(b"pdf")
    manual.write_bytes(b"pdf")
    (manuals / "readme.txt").write_text("not a PDF")

    assert find_pdfs(tmp_path) == [datasheet, manual]


def test_find_pdfs_rejects_duplicate_filenames(tmp_path):
    for group in ("Datasheets", "Manuals"):
        directory = tmp_path / group
        directory.mkdir()
        (directory / "shared.pdf").write_bytes(b"pdf")

    with pytest.raises(ValueError, match="filenames must be unique"):
        find_pdfs(tmp_path)


def test_ingest_pdfs_skips_existing_and_preserves_filenames(tmp_path):
    datasheets = tmp_path / "Datasheets"
    datasheets.mkdir()
    (datasheets / "already.pdf").write_bytes(b"pdf")
    new_pdf = datasheets / "New Data Sheet.pdf"
    new_pdf.write_bytes(b"pdf")
    session = Mock(spec=requests.Session)
    session.get.return_value = response_with_documents(["already.pdf"])
    session.post.return_value = response_with_documents([new_pdf.name])

    assert ingest_pdfs(tmp_path, "http://localhost:8102/v1/chatqna", session) == 0

    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args == ("http://localhost:8102/v1/chatqna/documents",)
    assert kwargs["files"]["files"][0] == new_pdf.name
    assert kwargs["files"]["files"][2] == "application/pdf"
    assert session.get.call_count == 1


def test_ingest_pdfs_continues_after_failed_upload(tmp_path):
    for name in ("one.pdf", "two.pdf"):
        (tmp_path / name).write_bytes(b"pdf")
    session = Mock(spec=requests.Session)
    session.get.return_value = response_with_documents([])
    session.post.side_effect = [requests.Timeout("upload timed out"), response_with_documents(["two.pdf"])]

    assert ingest_pdfs(tmp_path, "http://localhost:8102/v1/chatqna", session) == 1
    assert session.post.call_count == 2