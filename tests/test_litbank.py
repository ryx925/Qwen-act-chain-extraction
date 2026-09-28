import json

from trip_chain.data.litbank import LABELS, Span, sample
from trip_chain.evaluation.litbank import ground_prediction, score, valid_mentions


def test_sample_preserves_offsets_and_separate_annotation_layers():
    text = "Mary walked to London."
    spans = [
        Span("PER", 0, 4, "Mary"),
        Span("EVENT", 5, 11, "walked"),
        Span("GPE", 15, 21, "London"),
    ]
    row = sample("novel", 1, text, 0, len(text), spans)
    target = row["target"]
    assert target["evidence"] == "novel:c0001"
    assert target["EVENT"] == ["walked"]
    assert target["GPE"] == ["London"]
    assert row["spans"]["EVENT"] == [{"text": "walked", "start": 5, "end": 11}]
    assert target["FAC"] == []
    assert json.loads(row["completion"][0]["content"]) == target


def test_exact_mention_scoring_and_invalid_evidence():
    text = "Mary walked."
    target = {"evidence": "novel:c0001", **{label: [] for label in LABELS}}
    target["EVENT"] = ["walked"]
    gold = [{"id": "novel:c0001", "text": text, "target": target}]
    perfect = score(gold, [{"id": "novel:c0001", "raw": json.dumps(target)}])
    assert perfect["per_label"]["EVENT"]["f1"] == 1
    assert perfect["valid_json_rate"] == 1
    bad = dict(target, evidence="wrong")
    invalid = score(gold, [{"id": "novel:c0001", "raw": json.dumps(bad)}])
    assert invalid["valid_json_rate"] == 0
    assert invalid["json_parse_rate"] == 1
    assert invalid["per_label"]["EVENT"]["fn"] == 1
    assert invalid["diagnostic_mention_micro"]["f1"] == 1
    assert valid_mentions(target, text, "novel:c0001")["EVENT"] == {"walked": 1}


def test_repeated_mentions_are_counted():
    text = "She walked, then walked again."
    target = {"evidence": "novel:c0002", **{label: [] for label in LABELS}}
    target["EVENT"] = ["walked", "walked"]
    gold = [{"id": "novel:c0002", "text": text, "target": target}]
    predicted = dict(target, EVENT=["walked"])
    metrics = score(gold, [{"id": "novel:c0002", "raw": json.dumps(predicted)}])
    assert metrics["per_label"]["EVENT"]["tp"] == 1
    assert metrics["per_label"]["EVENT"]["fn"] == 1


def test_grounding_filters_absent_and_excess_mentions():
    raw = json.dumps({"EVENT": ["walked", "walked", "flew"], "PER": ["She"]})
    fixed = ground_prediction(raw, "She walked home.", "novel:c0003")
    assert fixed["evidence"] == "novel:c0003"
    assert fixed["EVENT"] == ["walked"]
    assert fixed["PER"] == ["She"]
    assert ground_prediction("{broken", "She walked home.", "novel:c0003")["EVENT"] == []
    partial = '{"EVENT":["walked"],"PER":["She","unfinished'
    recovered = ground_prediction(partial, "She walked home.", "novel:c0003")
    assert recovered["EVENT"] == ["walked"]
    assert recovered["PER"] == ["She"]
