from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FIELDS = (
    "origin",
    "destination",
    "departure_time",
    "arrival_time",
    "transport_mode",
    "purpose",
)


def normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def canonical_chain(record: dict[str, Any]) -> list[dict[str, Any]]:
    chain = record.get("trip_chain", [])
    return [
        {field: normalize(leg.get(field)) for field in FIELDS}
        for leg in sorted(chain, key=lambda item: item.get("sequence", 10**9))
    ]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_records(
    gold_records: Iterable[dict[str, Any]], prediction_records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {record["id"]: record for record in gold_records}
    pred_by_id = {record["id"]: record for record in prediction_records}
    field_counts = {field: [0, 0, 0] for field in FIELDS}
    exact = 0
    evaluated = 0

    for sample_id, gold in gold_by_id.items():
        predicted = pred_by_id.get(sample_id, {"trip_chain": []})
        gold_chain = canonical_chain(gold)
        pred_chain = canonical_chain(predicted)
        evaluated += 1
        exact += int(gold_chain == pred_chain)
        for index in range(max(len(gold_chain), len(pred_chain))):
            gold_leg = gold_chain[index] if index < len(gold_chain) else {}
            pred_leg = pred_chain[index] if index < len(pred_chain) else {}
            for field in FIELDS:
                gold_value = gold_leg.get(field)
                pred_value = pred_leg.get(field)
                tp, fp, fn = field_counts[field]
                if gold_value == pred_value and gold_value is not None:
                    tp += 1
                elif gold_value != pred_value:
                    fp += int(pred_value is not None)
                    fn += int(gold_value is not None)
                field_counts[field] = [tp, fp, fn]

    per_field = {field: _prf(*counts) for field, counts in field_counts.items()}
    totals = [sum(counts[i] for counts in field_counts.values()) for i in range(3)]
    return {
        "samples": evaluated,
        "missing_prediction_ids": sorted(set(gold_by_id) - set(pred_by_id)),
        "unexpected_prediction_ids": sorted(set(pred_by_id) - set(gold_by_id)),
        "chain_exact_match": exact / evaluated if evaluated else 0.0,
        "field_micro": _prf(*totals),
        "per_field": per_field,
    }


def parse_prediction(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError("prediction must be an object or JSON string")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("prediction JSON must be an object")
    return parsed


def read_jsonl(path: str | Path, predictions: bool = False) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if predictions and "prediction" in record:
                parsed = parse_prediction(record["prediction"])
                record = {"id": record["id"], **parsed}
            if "id" not in record or "trip_chain" not in record:
                raise ValueError(f"{path}:{line_number}: id and trip_chain are required")
            output.append(record)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate structured trip-chain predictions")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/evaluation.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_records(read_jsonl(args.gold), read_jsonl(args.predictions, predictions=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
