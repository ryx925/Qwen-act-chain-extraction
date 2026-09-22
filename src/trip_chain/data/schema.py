from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "trip-chain.schema.json"


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_record(record: dict[str, Any], schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> None:
    validator = Draft202012Validator(load_schema(schema_path))
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValueError(details)

    sequences = [leg["sequence"] for leg in record["trip_chain"]]
    expected = list(range(1, len(sequences) + 1))
    if sequences != expected:
        raise ValueError(f"trip_chain.sequence must be consecutive from 1; got {sequences}")
