"""Generate LitBank JSON and score exact grounded spans by label."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from trip_chain.data.litbank import LABELS


def valid_mentions(payload: object, text: str, evidence: str) -> dict[str, Counter[str]]:
    if not isinstance(payload, dict) or payload.get("evidence") != evidence:
        raise ValueError("Missing or incorrect evidence")
    if set(payload) != {"evidence", *LABELS}:
        raise ValueError("Incorrect JSON keys")
    found = {}
    for label in LABELS:
        values = payload[label]
        if not isinstance(values, list):
            raise TypeError(f"{label} is not an array")
        found[label] = Counter()
        for mention in values:
            if not isinstance(mention, str) or not mention:
                raise ValueError(f"Malformed {label} mention")
            found[label][mention] += 1
            if text.count(mention) < found[label][mention]:
                raise ValueError("Mention is absent or over-counted in input text")
    return found


def score(gold_records: list[dict], predictions: list[dict]) -> dict:
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    counts = {label: Counter() for label in LABELS}
    diagnostic_counts = {label: Counter() for label in LABELS}
    parsed_json = 0
    valid_json = 0
    exact_samples = 0
    failures = []
    for gold in gold_records:
        gold_spans = valid_mentions(gold["target"], gold["text"], gold["id"])
        predicted = by_id.get(gold["id"], {})
        try:
            raw = predicted["raw"]
            parsed = json.loads(raw)
            parsed_json += 1
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append({"id": gold["id"], "reason": str(exc)})
            parsed = None
        for label in LABELS:
            gold_texts = Counter(gold["target"][label])
            values = parsed.get(label) if isinstance(parsed, dict) else None
            predicted_texts = Counter(value for value in values if isinstance(value, str)) if isinstance(values, list) else Counter()
            diagnostic_counts[label]["tp"] += sum((predicted_texts & gold_texts).values())
            diagnostic_counts[label]["fp"] += sum((predicted_texts - gold_texts).values())
            diagnostic_counts[label]["fn"] += sum((gold_texts - predicted_texts).values())
        try:
            pred_spans = valid_mentions(parsed, gold["text"], gold["id"])
            valid_json += 1
        except (ValueError, TypeError) as exc:
            if parsed is not None:
                failures.append({"id": gold["id"], "reason": str(exc)})
            pred_spans = {label: Counter() for label in LABELS}
        exact_samples += all(pred_spans[label] == gold_spans[label] for label in LABELS)
        for label in LABELS:
            counts[label]["tp"] += sum((pred_spans[label] & gold_spans[label]).values())
            counts[label]["fp"] += sum((pred_spans[label] - gold_spans[label]).values())
            counts[label]["fn"] += sum((gold_spans[label] - pred_spans[label]).values())

    def rates(counter: Counter) -> dict:
        tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}

    aggregate = Counter()
    diagnostic_aggregate = Counter()
    for counter in counts.values():
        aggregate.update(counter)
    for counter in diagnostic_counts.values():
        diagnostic_aggregate.update(counter)
    return {
        "samples": len(gold_records),
        "json_parse_rate": round(parsed_json / len(gold_records), 4) if gold_records else 0.0,
        "valid_json": valid_json,
        "valid_json_rate": round(valid_json / len(gold_records), 4) if gold_records else 0.0,
        "exact_samples": exact_samples,
        "exact_sample_rate": round(exact_samples / len(gold_records), 4) if gold_records else 0.0,
        "per_label": {label: rates(counts[label]) for label in LABELS},
        "micro": rates(aggregate),
        "diagnostic_mention_micro": rates(diagnostic_aggregate),
        "failures": failures[:30],
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ground_prediction(raw: str, text: str, evidence: str) -> dict:
    """Make a usable JSON result using only mentions present in the supplied chunk."""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {}
        decoder = json.JSONDecoder()
        for label in LABELS:
            match = re.search(r'"' + label + r'"\s*:\s*\[', raw)
            if not match:
                continue
            values = []
            cursor = match.end()
            while cursor < len(raw):
                cursor = len(raw) - len(raw[cursor:].lstrip())
                if cursor >= len(raw) or raw[cursor] == "]":
                    break
                try:
                    value, end = decoder.raw_decode(raw, cursor)
                except ValueError:
                    break
                if not isinstance(value, str):
                    break
                values.append(value)
                cursor = end
                cursor = len(raw) - len(raw[cursor:].lstrip())
                if cursor >= len(raw) or raw[cursor] != ",":
                    break
                cursor += 1
            parsed[label] = values
    result = {"evidence": evidence, **{label: [] for label in LABELS}}
    for label in LABELS:
        values = parsed.get(label)
        if not isinstance(values, list):
            continue
        used = Counter()
        for mention in values:
            if isinstance(mention, str) and mention and used[mention] < text.count(mention):
                result[label].append(mention)
                used[mention] += 1
    return result


def generate(gold_records: list[dict], model_path: str, adapter_path: str, max_new_tokens: int, output_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization, device_map={"": 0}, dtype=torch.bfloat16, trust_remote_code=False)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    predictions = []
    from trip_chain.data.litbank import SYSTEM_PROMPT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(gold_records, 1):
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'evidence: {record["id"]}\ntext:\n{record["text"]}'},
            ]
            encoded = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            encoded = encoded.to(model.device)
            with torch.inference_mode():
                output = model.generate(encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True).strip()
            prediction = {"id": record["id"], "raw": raw}
            predictions.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0:
                print(f"Generated {index}/{len(gold_records)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LitBank EVENT and entity JSON extraction")
    parser.add_argument("--gold", type=Path, default=Path("data/processed/litbank_mentions/test_gold.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions/litbank_mentions_test.jsonl"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics/litbank_mentions_test.json"))
    parser.add_argument("--grounded-predictions", type=Path, default=Path("outputs/predictions/litbank_mentions_test_grounded.jsonl"))
    parser.add_argument("--grounded-metrics", type=Path, default=Path("outputs/metrics/litbank_mentions_test_grounded.json"))
    parser.add_argument("--model", default="/home/gollie/models/Qwen3-1.7B-modelscope")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-1.7b-litbank-mentions-qlora")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    gold_records = read_jsonl(args.gold)
    if args.limit:
        gold_records = gold_records[: args.limit]
    if not args.score_only:
        predictions = generate(gold_records, args.model, args.adapter, args.max_new_tokens, args.predictions)
    else:
        predictions = read_jsonl(args.predictions)
    metrics = score(gold_records, predictions)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_by_id = {row["id"]: row["raw"] for row in predictions}
    grounded = [
        {"id": gold["id"], "raw": json.dumps(ground_prediction(prediction_by_id.get(gold["id"], ""), gold["text"], gold["id"]), ensure_ascii=False, separators=(",", ":"))}
        for gold in gold_records
    ]
    args.grounded_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.grounded_predictions.open("w", encoding="utf-8") as handle:
        for row in grounded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    grounded_metrics = score(gold_records, grounded)
    args.grounded_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.grounded_metrics.write_text(json.dumps(grounded_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
