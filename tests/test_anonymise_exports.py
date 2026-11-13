"""Tests for code/annotation/anonymise_exports.py."""

from __future__ import annotations

from pathlib import Path

from code.annotation.anonymise_exports import (
    LABELS_DIR,
    anonymise_bytes,
    find_addresses,
    main,
    pseudonym_for,
)


def test_pseudonym_follows_annotator_number() -> None:
    assert pseudonym_for(Path("annotator1_raw.json")) == b"A1@anonymised.invalid"
    assert pseudonym_for(Path("round2/annotator2_raw.json")) == b"A2@anonymised.invalid"


def test_only_addresses_change() -> None:
    raw = b'[{"completed_by":{"id":3,"email":"someone.x@gmail.com"}},{"email":"someone.x@gmail.com"}]'
    new, count = anonymise_bytes(raw, b"A1@anonymised.invalid")
    assert count == 2
    assert new == raw.replace(b"someone.x@gmail.com", b"A1@anonymised.invalid")
    assert find_addresses(new) == []


def test_anonymising_twice_changes_nothing() -> None:
    once, _ = anonymise_bytes(b'{"email":"a@b.org"}', b"A2@anonymised.invalid")
    twice, count = anonymise_bytes(once, b"A2@anonymised.invalid")
    assert twice == once and count == 0


def test_repository_exports_hold_no_personal_addresses() -> None:
    assert LABELS_DIR.is_dir()
    assert main(["--check"]) == 0
