from __future__ import annotations

import pytest

from trip_chain.evaluation.metrics import evaluate_records


def leg(origin: str, destination: str) -> dict:
    return {
        "sequence": 1,
        "origin": origin,
        "destination": destination,
        "departure_time": None,
        "arrival_time": None,
        "transport_mode": "地铁",
        "purpose": "上班",
    }


def test_perfect_prediction() -> None:
    gold = [{"id": "1", "trip_chain": [leg("家", "公司")]}]
    report = evaluate_records(gold, gold)
    assert report["chain_exact_match"] == 1.0
    assert report["field_micro"]["f1"] == 1.0


def test_wrong_destination_counts_fp_and_fn() -> None:
    gold = [{"id": "1", "trip_chain": [leg("家", "公司")]}]
    predictions = [{"id": "1", "trip_chain": [leg("家", "学校")]}]
    report = evaluate_records(gold, predictions)
    destination = report["per_field"]["destination"]
    assert destination["fp"] == 1
    assert destination["fn"] == 1
    assert report["chain_exact_match"] == 0.0
    assert report["field_micro"]["f1"] == pytest.approx(0.75)
