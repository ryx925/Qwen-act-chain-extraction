from __future__ import annotations

import json
from pathlib import Path

import pytest

from trip_chain.data.schema import validate_record

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "samples" / "annotated_examples.jsonl"


def test_all_examples_match_schema() -> None:
    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        validate_record(json.loads(line))


def test_sequence_must_be_consecutive() -> None:
    record = json.loads(SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0])
    record["trip_chain"][0]["sequence"] = 2
    with pytest.raises(ValueError, match="consecutive"):
        validate_record(record)
