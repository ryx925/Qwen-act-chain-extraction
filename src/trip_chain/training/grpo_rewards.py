"""Six deterministic rewards; KL and policy optimization belong to TRL."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from trip_chain.data.grpo import (
    PERSON_FIELDS,
    evidence_valid,
    strict_json,
    structure_valid,
    validate_shape,
)

REWARD_NAMES = ("node", "edge", "time", "chain", "format", "unsupported")


def norm(value: Any) -> Any:
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def assignment(scores: list[list[float]]) -> dict[int, int]:
    """Maximum-weight one-to-one assignment; reject zero-similarity matches."""
    if not scores or not scores[0]:
        return {}
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(scores, maximize=True)
    return {int(i): int(j) for i, j in zip(rows, cols, strict=True) if scores[i][j] > 0}


def node_similarity(p: dict, g: dict) -> float:
    score = sum(
        weight for field, weight in (
            ("activity_name", 4), ("activity_place", 3),
            ("activity_start_time", 1), ("activity_duration", 1),
        ) if p[field] is not None and norm(p[field]) == norm(g[field])
    )
    fields = ("activity_name", "activity_place", "activity_start_time", "activity_duration")
    return score or float(all(p[f] is None and g[f] is None for f in fields))


def person_similarity(p: dict, g: dict) -> float:
    a, b = p["person"], g["person"]
    if a["person_name"] and b["person_name"]:
        if norm(a["person_name"]) != norm(b["person_name"]):
            return 0
        return 20 + sum(
            a[f] is not None and norm(a[f]) == norm(b[f]) for f in PERSON_FIELDS[1:]
        )
    score = sum(
        5 for f in PERSON_FIELDS[1:] if a[f] is not None and norm(a[f]) == norm(b[f])
    )
    if all(a[f] is None and b[f] is None for f in PERSON_FIELDS):
        score = 1
    # Anonymous/partially identified people can still be aligned by their activities.
    return score + max(
        (node_similarity(x, y) for x in p["nodes"] for y in g["nodes"]), default=0
    ) / 10


def f1(pred: Counter, gold: Counter, *, empty_score: float = 0.0) -> float:
    total = pred.total() + gold.total()
    return 2 * (pred & gold).total() / total if total else empty_score


def facts(chains: list[dict], mapping: dict | None = None) -> dict[str, Counter]:
    """Include person/node/edge identities so duplication and wrong owners cost recall/precision."""
    result = {name: Counter() for name in ("node", "edge", "time", "chain")}
    for ci, chain in enumerate(chains):
        cid, node_map = (("g", ci), None) if mapping is None else mapping[ci]
        result["node"][("person", cid)] += 1
        result["chain"][("person", cid)] += 1
        for field in PERSON_FIELDS:
            if chain["person"][field] is not None:
                result["node"][("person_field", cid, field, norm(chain["person"][field]))] += 1
        node_ids = []
        sequence_ids: dict[int, Any] = {}
        for ni, node in enumerate(chain["nodes"]):
            nid = ("g", ni) if node_map is None else node_map.get(ni, ("p", ni))
            node_ids.append(nid)
            # Duplicate sequence values must not collapse distinct predicted nodes.
            if node["sequence"] in sequence_ids:
                sequence_ids[node["sequence"]] = ("duplicate", node["sequence"])
            else:
                sequence_ids[node["sequence"]] = nid
            result["node"][("node", cid, nid)] += 1
            result["chain"][("node", cid, nid)] += 1
            for field in ("activity_name", "activity_place"):
                if node[field] is not None:
                    result["node"][("node_field", cid, nid, field, norm(node[field]))] += 1
            for field in ("activity_start_time", "activity_duration"):
                if node[field] is not None:
                    result["time"][("node_time", cid, nid, field, norm(node[field]))] += 1
        for i, left in enumerate(node_ids):
            for right in node_ids[i + 1:]:
                result["chain"][("before", cid, left, right)] += 1
        for edge in chain["edges"]:
            left = sequence_ids.get(edge["from_sequence"], ("missing", edge["from_sequence"]))
            right = sequence_ids.get(edge["to_sequence"], ("missing", edge["to_sequence"]))
            key = (cid, left, right)
            result["edge"][("edge", *key)] += 1
            if edge["travel_mode"] is not None:
                result["edge"][("mode", *key, norm(edge["travel_mode"]))] += 1
            if edge["travel_duration"] is not None:
                result["time"][("travel_time", *key, norm(edge["travel_duration"]))] += 1
    return result


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1:
        message = completion[0]
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError("Expected a completion string or one assistant message")


@lru_cache(maxsize=256)
def score_completion(completion: str, gold_json: str, source_text: str) -> dict[str, float]:
    # A broken reward-side gold record is a programming/data error, never a model penalty.
    gold = strict_json(gold_json)
    validate_shape(gold)
    scores = dict.fromkeys(REWARD_NAMES, 0.0)
    try:
        prediction = strict_json(completion)
        validate_shape(prediction)
    except (ValueError, TypeError, KeyError, RecursionError):
        return {**scores, "unsupported": -1.0}
    pc, gc = prediction["chains"], gold["chains"]
    people = assignment([[person_similarity(p, g) for g in gc] for p in pc])
    mapping = {}
    for i, chain in enumerate(pc):
        if i not in people:
            mapping[i] = (("p", i), {})
        else:
            gi = people[i]
            aligned = assignment([
                [node_similarity(p, g) for g in gc[gi]["nodes"]] for p in chain["nodes"]
            ])
            mapping[i] = (("g", gi), {p: ("g", g) for p, g in aligned.items()})
    pf, gf = facts(pc, mapping), facts(gc)
    scores["node"] = f1(pf["node"], gf["node"], empty_score=1.0)
    scores["edge"] = f1(pf["edge"], gf["edge"])
    scores["time"] = f1(pf["time"], gf["time"])
    structure = sum(structure_valid(c) for c in pc) / len(pc) if pc else 1.0
    scores["chain"] = f1(pf["chain"], gf["chain"], empty_score=1.0) * structure
    invalid_evidence = sum(
        not evidence_valid(item["evidence"], source_text)
        for c in pc for item in c["nodes"] + c["edges"] if "evidence" in item
    )
    scores["format"] = float(invalid_evidence == 0)
    # Gold may use normalized activity labels. Compare labels to gold, not to raw substrings.
    # This penalizes fabricated values, null->value filling and unmatched extra structures.
    predicted = pf["node"] + pf["edge"] + pf["time"]
    expected = gf["node"] + gf["edge"] + gf["time"]
    wrong = predicted.total() - (predicted & expected).total() + invalid_evidence
    scores["unsupported"] = -wrong / max(1, predicted.total() + invalid_evidence)
    return scores


def build_reward_functions(trace_path: Path | None = None) -> list[Callable]:
    """TRL repeats metadata per candidate; fail on misalignment instead of truncating zip."""
    def create(name: str) -> Callable:
        def reward(
            completions: list, gold_json: list[str], source_text: list[str], **kwargs: Any
        ) -> list[float]:
            if not (len(completions) == len(gold_json) == len(source_text)):
                raise ValueError("Reward inputs have different lengths")
            result = []
            trace = []
            for i, (completion, gold, text) in enumerate(
                zip(completions, gold_json, source_text, strict=True)
            ):
                raw = completion_text(completion)
                scores = score_completion(raw, gold, text)
                result.append(scores[name])
                if trace_path is not None and name == REWARD_NAMES[-1]:
                    state = kwargs.get("trainer_state")
                    trace.append({
                        "step": getattr(state, "global_step", None),
                        "sample_id": kwargs.get("sample_id", [None] * len(completions))[i],
                        "completion": raw, "rewards": scores,
                    })
            if trace:
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                with trace_path.open("a", encoding="utf-8") as handle:
                    for item in trace:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            return result

        reward.__name__ = f"reward_{name}"
        return reward

    return [create(name) for name in REWARD_NAMES]
