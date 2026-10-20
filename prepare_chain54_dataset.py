"""Archive and project the approved + two synthetic batches into 54 training records."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trip_chain.data.grpo import (
    EDGE_FIELDS,
    NODE_FIELDS,
    PERSON_FIELDS,
    evidence_valid,
    load_annotations,
    strict_json,
    validate_shape,
    write_json,
)

OUTPUT = ROOT / "data/annotations/chain54_20261020"
SOURCES = [
    ("human_approved", ROOT.parents[1] / "GLiNER2/annotation/人工已审核/approved_annotations.jsonl"),
    ("synthetic_20261015", ROOT.parents[1] / "GLiNER2/annotation/编造文本_出行链_2026-10-15/synthetic_travel_chains.jsonl"),
    ("synthetic_20261020", ROOT.parents[1] / "GLiNER2/annotation/编造文本_出行链_2026-10-20/synthetic_travel_chains.jsonl"),
]


def project_record(record):
    """Keep label values verbatim; evidence-only extensions stay in archived originals."""
    result = {**record, "chains": []}
    for chain in record["chains"]:
        if set(chain) - {"person", "person_evidence", "nodes", "edges"}:
            raise ValueError("Unknown chain extension; inspect instead of silently dropping it")
        if set(chain["person"]) != set(PERSON_FIELDS):
            raise ValueError("Unknown person fields")
        projected = {"person": dict(chain["person"]), "nodes": [], "edges": []}
        for key, fields in (("nodes", NODE_FIELDS), ("edges", EDGE_FIELDS)):
            for item in chain[key]:
                if set(item) - set(fields) - {"evidence", "field_evidence"}:
                    raise ValueError(f"Unknown {key} extension")
                target = {field: item[field] for field in fields}
                if "evidence" in item:
                    if not evidence_valid(item["evidence"], record["text"]):
                        raise ValueError("Source evidence does not match original passage")
                    target["evidence"] = item["evidence"]
                projected[key].append(target)
        result["chains"].append(projected)
    validate_shape({"chains": result["chains"]})
    return result


def prepare():
    rows, manifest = [], []
    for label, source in SOURCES:
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        records = [strict_json(s) for s in raw.decode("utf-8-sig").splitlines() if s.strip()]
        for record in records:
            projected = project_record(record)
            projected["dataset_batch"] = label
            projected["source_annotation_sha256"] = digest
            rows.append(projected)
        manifest.append({"batch": label, "source": str(source), "rows": len(records),
                         "sha256": digest, "archive": f"originals/{label}.jsonl"})
    if len(rows) != 54 or len({r["chunk_id"] for r in rows}) != 54:
        raise ValueError("Expected 54 distinct annotated records")
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    target = OUTPUT / "all_annotations.jsonl"
    if target.exists() and target.read_text(encoding="utf-8") != payload:
        raise ValueError("Existing 54-record snapshot differs; create a new version")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "originals").mkdir(exist_ok=True)
    for (label, source), item in zip(SOURCES, manifest, strict=True):
        destination = OUTPUT / item["archive"]
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Original archive changed: {label}")
        shutil.copy2(source, destination)
    target.write_text(payload, encoding="utf-8")
    validated = load_annotations([target], allow_synthetic=True)
    summary = {"samples": len(validated), "human_approved": 14, "synthetic": 40,
               "chains": sum(len(r["chains"]) for r in rows),
               "nodes": sum(len(c["nodes"]) for r in rows for c in r["chains"]),
               "edges": sum(len(c["edges"]) for r in rows for c in r["chains"]),
               "sources": manifest, "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "projection": "Only person_evidence and field_evidence extensions omitted from trainer labels; originals archived verbatim"}
    write_json(OUTPUT / "manifest.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=True, indent=2))
