from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trip_chain.data.schema import validate_record

SYSTEM_PROMPT = (
    "你是出行链信息抽取器。只依据用户提供的文本抽取按时间排序的 trip_chain。"
    "输出单个合法 JSON 对象，不要解释，不要 Markdown。缺失信息使用 null，禁止猜测。"
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"No valid records found in {path}")
    return records


def to_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    target = {"trip_chain": record["trip_chain"]}
    return {
        "id": record["id"],
        "source_id": record["source_id"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["text"]},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            }
        ],
    }


def grouped_split(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Ratios must satisfy 0 < train_ratio < 1 and 0 <= validation_ratio < 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["source_id"]].append(record)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    train_end = max(1, round(count * train_ratio))
    validation_end = max(train_end + 1, round(count * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, count)

    key_splits = {
        "train": keys[:train_end],
        "validation": keys[train_end:validation_end],
        "test": keys[validation_end:],
    }
    return {
        name: [record for key in split_keys for record in groups[key]]
        for name, split_keys in key_splits.items()
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_dataset(
    input_path: str | Path, output_dir: str | Path, seed: int = 42
) -> dict[str, int]:
    records = read_jsonl(input_path)
    splits = grouped_split(records, seed=seed)
    output = Path(output_dir)
    for name, split_records in splits.items():
        write_jsonl(output / f"{name}.jsonl", map(to_sft_record, split_records))
        if name == "test":
            write_jsonl(output / "test_gold.jsonl", split_records)
    return {name: len(split_records) for name, split_records in splits.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and split trip-chain JSONL data")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = prepare_dataset(args.input, args.output_dir, args.seed)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
