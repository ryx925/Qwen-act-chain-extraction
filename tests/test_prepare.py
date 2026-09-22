from __future__ import annotations

from trip_chain.data.prepare import grouped_split, to_sft_record


def make_record(index: int, source_id: str) -> dict:
    return {
        "id": f"id-{index}",
        "source_id": source_id,
        "text": "从家到公司。",
        "trip_chain": [
            {
                "sequence": 1,
                "origin": "家",
                "destination": "公司",
                "departure_time": None,
                "arrival_time": None,
                "transport_mode": None,
                "purpose": "上班",
            }
        ],
    }


def test_grouped_split_has_no_source_leakage() -> None:
    records = [make_record(i, f"source-{i // 2}") for i in range(20)]
    splits = grouped_split(records)
    groups = [{record["source_id"] for record in split} for split in splits.values()]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_sft_target_contains_only_trip_chain() -> None:
    converted = to_sft_record(make_record(1, "source-1"))
    assert converted["prompt"][0]["role"] == "system"
    assert converted["completion"][0]["content"].startswith('{"trip_chain":')
