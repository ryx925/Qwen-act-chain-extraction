"""Convert LitBank BRAT spans to grounded, length-bounded JSON SFT samples."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

LABELS = ("EVENT", "PER", "FAC", "GPE", "LOC", "VEH")
SYSTEM_PROMPT = (
    "Extract LitBank annotations from the supplied English text. EVENT means an asserted, "
    "actually occurring event, not a hypothetical, future, or generic event. Extract entity "
    "mentions under PER, FAC, GPE, LOC, and VEH. Return one JSON object with exactly "
    "evidence, EVENT, PER, FAC, GPE, LOC, VEH. Each label contains an array of exact text "
    "mentions in source order; repeat a string when it is annotated more than once. "
    "Keep nested mentions. Use empty arrays when absent. Do not infer event-entity relations. "
    "Output JSON only."
)


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    text: str


def read_spans(path: Path, text: str, labels: set[str]) -> list[Span]:
    spans: list[Span] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith("T"):
            continue  # BRAT annotator notes are not text annotations.
        label, offsets = fields[1].split(" ", 1)
        if label not in labels:
            continue
        pieces = [tuple(map(int, piece.split())) for piece in offsets.split(";")]
        observed = " ".join(text[start:end] for start, end in pieces)
        if observed != fields[2]:
            raise ValueError(f"Offset/text mismatch in {path}: {fields[0]}")
        if len(pieces) != 1:
            raise ValueError(f"Discontinuous annotation in {path}: {fields[0]}")
        start, end = pieces[0]
        spans.append(Span(label, start, end, fields[2]))
    return sorted(spans, key=lambda item: (item.start, item.end, item.label))


def safe_units(text: str, spans: list[Span], tokenizer, max_input_tokens: int) -> list[tuple[int, int]]:
    """Prefer line boundaries; split a long line at safe whitespace boundaries."""
    line_ends = [match.end() for match in re.finditer(r"[^\n]*\n|[^\n]+$", text)]
    units: list[tuple[int, int]] = []
    start = 0
    for line_end in line_ends:
        while start < line_end:
            if len(tokenizer.encode(text[start:line_end], add_special_tokens=False)) <= max_input_tokens:
                units.append((start, line_end))
                start = line_end
                break
            candidates = [match.end() for match in re.finditer(r"\s+", text[start:line_end])]
            best = None
            for relative_end in candidates:
                end = start + relative_end
                if any(span.start < end < span.end for span in spans):
                    continue
                if len(tokenizer.encode(text[start:end], add_special_tokens=False)) > max_input_tokens:
                    break
                best = end
            if best is None:
                raise ValueError(f"Cannot safely split text near character {start}")
            units.append((start, best))
            start = best
    return units


def sample(document_id: str, index: int, text: str, start: int, end: int, spans: list[Span]):
    chunk_id = f"{document_id}:c{index:04d}"
    chunk_text = text[start:end]
    target = {"evidence": chunk_id, **{label: [] for label in LABELS}}
    gold_spans = {label: [] for label in LABELS}
    for span in spans:
        if start <= span.start and span.end <= end:
            target[span.label].append(span.text)
            gold_spans[span.label].append(
                {"text": span.text, "start": span.start - start, "end": span.end - start}
            )
        elif span.start < end and span.end > start:
            raise ValueError(f"Annotation crosses chunk boundary: {document_id} {span}")
    user = f"evidence: {chunk_id}\ntext:\n{chunk_text}"
    return {
        "id": chunk_id,
        "source_id": document_id,
        "start": start,
        "end": end,
        "text": chunk_text,
        "target": target,
        "spans": gold_spans,
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}],
    }


def token_counts(record: dict, tokenizer) -> dict[str, int]:
    prompt = sum(len(tokenizer.encode(part["content"], add_special_tokens=False)) for part in record["prompt"])
    output = len(tokenizer.encode(record["completion"][0]["content"], add_special_tokens=False))
    source = len(tokenizer.encode(record["text"], add_special_tokens=False))
    # Leave room for chat delimiters, assistant header, and EOS.
    return {"prompt": prompt - source, "input": source, "output": output, "total_upper": prompt + output + 48}


def convert_document(event_path: Path, entity_path: Path, tokenizer, max_length: int, max_input: int):
    text = event_path.with_suffix(".txt").read_text(encoding="utf-8")
    entity_text = entity_path.with_suffix(".txt").read_text(encoding="utf-8")
    if text != entity_text:
        raise ValueError(f"Event/entity text differs: {event_path.name}")
    spans = read_spans(event_path, text, {"EVENT"}) + read_spans(entity_path, text, set(LABELS) - {"EVENT"})
    spans.sort(key=lambda item: (item.start, item.end, item.label))
    units = safe_units(text, spans, tokenizer, max_input)
    document_id = event_path.stem.removesuffix("_brat")
    records = []
    cursor = 0
    while cursor < len(units):
        best = None
        for next_cursor in range(cursor + 1, len(units) + 1):
            start, end = units[cursor][0], units[next_cursor - 1][1]
            if len(tokenizer.encode(text[start:end], add_special_tokens=False)) > max_input:
                break
            candidate = sample(document_id, len(records) + 1, text, start, end, spans)
            if token_counts(candidate, tokenizer)["total_upper"] > max_length:
                break
            best = (next_cursor, candidate)
        if best is None:
            raise ValueError(f"Single unit exceeds token budget in {event_path.name} at {units[cursor]}")
        next_cursor, record = best
        records.append(record)
        cursor = next_cursor
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LitBank EVENT and entity SFT data")
    parser.add_argument("--litbank-dir", type=Path, default=Path("litbank"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/litbank_mentions"))
    parser.add_argument("--tokenizer", default="outputs/checkpoints/smoke-test")
    parser.add_argument("--max-length", type=int, default=832)
    parser.add_argument("--max-input", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    events_dir = args.litbank_dir / "events" / "brat"
    entities_dir = args.litbank_dir / "entities" / "brat"
    event_files = sorted(events_dir.glob("*.ann"))
    if len(event_files) != 100:
        raise ValueError(f"Expected 100 LitBank event documents, found {len(event_files)}")
    documents = []
    for path in event_files:
        entity_path = entities_dir / path.name
        if not entity_path.exists():
            raise FileNotFoundError(entity_path)
        documents.append((path, entity_path))
    random.Random(args.seed).shuffle(documents)
    splits = {"train": documents[:80], "validation": documents[80:90], "test": documents[90:]}
    report = {"documents": {}, "samples": {}, "tokens": {}}
    for name, group in splits.items():
        records = [record for paths in group for record in convert_document(*paths, tokenizer, args.max_length, args.max_input)]
        write_jsonl(args.output_dir / f"{name}.jsonl", [
            {key: value for key, value in record.items() if key in {"id", "source_id", "prompt", "completion"}}
            for record in records
        ])
        write_jsonl(args.output_dir / f"{name}_gold.jsonl", [
            {key: value for key, value in record.items() if key not in {"prompt", "completion"}}
            for record in records
        ])
        report["documents"][name] = len(group)
        report["samples"][name] = len(records)
        counts = [token_counts(record, tokenizer) for record in records]
        report["tokens"][name] = {
            field: {"min": min(row[field] for row in counts), "median": sorted(row[field] for row in counts)[len(counts)//2], "max": max(row[field] for row in counts)}
            for field in ("prompt", "input", "output", "total_upper")
        }
    (args.output_dir / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
