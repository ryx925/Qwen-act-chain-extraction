from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from trip_chain.data.grpo import (
    load_annotations,
    make_training_row,
    split_records,
    strict_json,
    target_without_evidence,
    validate_shape,
)
from trip_chain.training.grpo_rewards import (
    REWARD_NAMES,
    build_reward_functions,
    score_completion,
)
from trip_chain.training.train_grpo import (
    build_parser,
    load_config,
    tokenize_splits,
    trainer_kwargs,
)


@pytest.fixture
def gold() -> dict:
    return {"chains": [{
        "person": {"person_name": "Marta", "identity_occupation": "worker", "age": None},
        "nodes": [
            {"activity_name": "wash cloth", "activity_place": "river",
             "activity_start_time": "daybreak", "activity_duration": None, "sequence": 1},
            {"activity_name": "sell cloth", "activity_place": "market",
             "activity_start_time": None, "activity_duration": "an hour", "sequence": 2},
        ],
        "edges": [{"from_sequence": 1, "to_sequence": 2,
                   "travel_mode": "walk", "travel_duration": "ten minutes"}],
    }]}


def score(pred: dict | str, gold: dict) -> dict:
    raw = json.dumps(pred) if isinstance(pred, dict) else pred
    return score_completion(raw, json.dumps(gold), "Marta washes cloth, then sells it.")


def test_perfect_prediction_and_normalized_gold_labels(gold: dict) -> None:
    result = score(gold, gold)
    assert result == {**dict.fromkeys(REWARD_NAMES[:-1], 1.0), "unsupported": 0.0}


def test_filling_unknown_field_is_penalized(gold: dict) -> None:
    pred = copy.deepcopy(gold)
    pred["chains"][0]["person"]["age"] = "twenty"
    result = score(pred, gold)
    assert result["unsupported"] < 0
    assert result["node"] < 1


def test_empty_output_cannot_get_content_reward_on_positive(gold: dict) -> None:
    result = score({"chains": []}, gold)
    assert all(result[k] == 0 for k in ("node", "edge", "time", "chain"))
    assert result["format"] == 1


def test_correct_negative_and_person_only() -> None:
    empty = {"chains": []}
    assert score(empty, empty) == {
        "node": 1, "edge": 0, "time": 0, "chain": 1, "format": 1, "unsupported": 0
    }
    person_only = {"chains": [{"person": {
        "person_name": "Marta", "identity_occupation": None, "age": None
    }, "nodes": [], "edges": []}]}
    assert score(person_only, person_only)["node"] == 1
    assert score(empty, person_only)["node"] == 0


@pytest.mark.parametrize("bad", [
    '{"chains":[', '```json\n{"chains":[]}\n```',
    '{"chains":[],"chains":[]}', '{"chains":NaN}', '{"chains":{},"extra":0}',
])
def test_malformed_outputs_are_not_repaired_or_rewarded(bad: str, gold: dict) -> None:
    result = score(bad, gold)
    assert result["format"] == 0
    assert result["unsupported"] == -1
    assert sum(result.values()) == -1


def test_duplicate_chain_cannot_double_reward(gold: dict) -> None:
    pred = copy.deepcopy(gold)
    pred["chains"].append(copy.deepcopy(pred["chains"][0]))
    result = score(pred, gold)
    assert result["node"] < 1
    assert result["edge"] < 1
    assert result["unsupported"] < 0


def test_reversed_order_and_invalid_connectivity(gold: dict) -> None:
    pred = copy.deepcopy(gold)
    chain = pred["chains"][0]
    chain["nodes"].reverse()
    for i, node in enumerate(chain["nodes"], 1):
        node["sequence"] = i
    result = score(pred, gold)
    assert result["node"] == 1  # Same nodes, different order.
    assert result["chain"] < 1
    assert result["edge"] == 0
    chain["edges"][0]["to_sequence"] = 99
    assert score(pred, gold)["chain"] == 0


def test_people_order_is_irrelevant_but_swapping_owners_is_wrong(gold: dict) -> None:
    g = copy.deepcopy(gold)
    other = copy.deepcopy(g["chains"][0])
    other["person"]["person_name"] = "Lucia"
    other["nodes"][0]["activity_name"] = "read"
    other["nodes"][1]["activity_place"] = "school"
    g["chains"].append(other)
    p = copy.deepcopy(g)
    p["chains"].reverse()
    assert score(p, g)["node"] == 1
    p["chains"][0]["nodes"], p["chains"][1]["nodes"] = (
        p["chains"][1]["nodes"], p["chains"][0]["nodes"]
    )
    assert score(p, g)["node"] < 1


def test_activity_and_travel_duration_are_distinct(gold: dict) -> None:
    p = copy.deepcopy(gold)
    c = p["chains"][0]
    c["nodes"][1]["activity_duration"], c["edges"][0]["travel_duration"] = (
        c["edges"][0]["travel_duration"], c["nodes"][1]["activity_duration"]
    )
    assert score(p, gold)["time"] < 1


def test_bad_evidence_and_bool_sequence(gold: dict) -> None:
    p = copy.deepcopy(gold)
    p["chains"][0]["nodes"][0]["evidence"] = {"text": "wrong", "start": 0, "end": 5}
    assert score(p, gold)["format"] == 0
    p["chains"][0]["nodes"][0]["sequence"] = True
    with pytest.raises(ValueError):
        validate_shape(p)


def test_unknown_boundary_endpoint_is_preserved(gold: dict) -> None:
    from trip_chain.data.grpo import structure_valid

    chain = gold["chains"][0]
    chain["edges"].insert(0, {"from_sequence": None, "to_sequence": 1,
                              "travel_mode": None, "travel_duration": None})
    validate_shape(gold)
    assert structure_valid(chain)
    assert score(gold, gold)["edge"] == 1
    assert score(gold, gold)["chain"] == 1
    chain["edges"][0]["to_sequence"] = 2
    assert not structure_valid(chain)


def test_reward_metadata_alignment_and_trace(tmp_path: Path, gold: dict) -> None:
    trace = tmp_path / "trace.jsonl"
    funcs = build_reward_functions(trace)
    kw = {"completions": [json.dumps(gold), '{"chains":[]}'],
          "gold_json": [json.dumps(gold)] * 2, "source_text": ["text"] * 2,
          "sample_id": ["a", "a"]}
    assert [f.__name__ for f in funcs] == [f"reward_{n}" for n in REWARD_NAMES]
    for f in funcs:
        assert len(f(**kw)) == 2
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError):
        funcs[0](completions=["{}"], gold_json=[], source_text=[])


def test_split_keeps_source_chapters_and_people_together() -> None:
    records = []
    for i in range(12):
        records.append({"chunk_id": str(i), "split_group": f"chapter-{i // 2}",
                        "chains": [{"person": {"person_name": f"person-{i // 4}"},
                                    "nodes": []}]})
    splits = split_records(records, 0.34)
    a, b = splits["train"], splits["validation"]
    assert a and b
    assert {r["split_group"] for r in a}.isdisjoint(r["split_group"] for r in b)
    assert {r["chains"][0]["person"]["person_name"] for r in a}.isdisjoint(
        r["chains"][0]["person"]["person_name"] for r in b
    )


def test_loader_rejects_unreviewed_and_conflicting_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    base = {"chunk_id": "a", "text": "a passage", "chains": [], "status": "needs_review"}
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="human_approved"):
        load_annotations([path])
    base["status"] = "human_approved"
    second = {**base, "chunk_id": "b", "chains": [{"person": {
        "person_name": "Marta", "identity_occupation": None, "age": None
    }, "nodes": [], "edges": []}]}
    path.write_text(json.dumps(base) + "\n" + json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_annotations([path])


def test_synthetic_requires_explicit_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "synth.jsonl"
    path.write_text(json.dumps({"chunk_id": "a", "text": "passage", "chains": [],
                                "status": "synthetic_labeled", "synthetic": True}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="include-synthetic"):
        load_annotations([path])
    assert len(load_annotations([path], allow_synthetic=True)) == 1


class TokenizerStub:
    def apply_chat_template(self, messages: list, **kwargs: object) -> str:
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"] + messages[1]["content"]

    def __call__(self, text: str, **kwargs: object) -> dict:
        return {"input_ids": text.split()}


def test_gold_never_enters_policy_prompt_and_no_silent_truncation(gold: dict) -> None:
    record = {"chunk_id": "a", "text": "UNIQUE_PASSAGE", "chains": gold["chains"]}
    row = make_training_row(record, TokenizerStub())
    assert "UNIQUE_PASSAGE" in row["prompt"]
    assert "Marta" not in row["prompt"]
    assert "Marta" in row["gold_json"]
    assert "completion" not in row
    with pytest.raises(ValueError, match="never be silently truncated"):
        tokenize_splits({"train": [record]}, TokenizerStub(),
                        {"max_prompt_length": 1, "max_completion_length": 1})


def test_config_enables_kl_and_six_equal_weights() -> None:
    cfg = load_config(build_parser().parse_args([]))
    values = trainer_kwargs(cfg["training"], cfg["reward_weights"], bf16=True)
    assert values["beta"] > 0
    assert values["reward_weights"] == [1] * 6
    assert values["loss_type"] == "grpo"
    assert values["scale_rewards"] == "group"
    assert "max_prompt_length" not in values
    assert values["generation_batch_size"] % values["num_generations"] == 0
    assert target_without_evidence([]) == {"chains": []}
    with pytest.raises(ValueError):
        strict_json('{"a":1,"a":2}')


def test_smoke_has_nonzero_starting_learning_rate() -> None:
    cfg = load_config(build_parser().parse_args(["--smoke-test"]))
    assert cfg["training"]["max_steps"] == 1
    assert cfg["training"]["warmup_steps"] == 0
    assert cfg["training"]["learning_rate"] > 0


def test_stratification_keeps_positive_groups_on_both_sides() -> None:
    records = [{"chunk_id": str(i), "split_group": str(i),
                "chains": [{"person": {"person_name": str(i)},
                            "nodes": [1] if i < 3 else []}]} for i in range(12)]
    splits = split_records(records, 0.2)
    assert all(any(c["nodes"] for r in split for c in r["chains"])
               for split in splits.values())
