from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from trip_chain.training.grpo_microbatch import MicroBatchGenerationMixin
from trip_chain.training.grpo_plots import reward_rows
from trip_chain.training.train_grpo import (
    PROJECT_ROOT,
    build_parser,
    load_config,
    prepare,
    run_contract,
)


class FakeGeneration:
    def __init__(self):
        self.use_vllm = self.use_transformers_paged = False
        self.state = SimpleNamespace(global_step=7)
        self.calls = []
        self.rng = random.Random(42)

    def _generate_single_turn(self, ids, images, fields):
        self.calls.append(ids)
        return [[p[0], self.rng.randrange(1000000)] for p in ids], None, {}


class Chunked(MicroBatchGenerationMixin, FakeGeneration):
    pass


@pytest.mark.parametrize("size", [1, 2, 4])
def test_chunks_preserve_all_six_candidates_order_rng_and_policy(size, tmp_path):
    path = tmp_path / "micro.jsonl"
    trainer = Chunked(generation_micro_batch_size=size, generation_trace_path=path)
    ids = [[101]] * 6 + [[202]] * 6
    outputs, logps, extra = trainer._generate_single_turn(ids, None, {})
    reference, _, _ = FakeGeneration()._generate_single_turn(ids, None, {})
    assert outputs == reference  # No hidden reseeding, reordering, or filtering.
    assert len(outputs) == 12
    assert max(len(call) for call in trainer.calls) <= size
    assert trainer.state.global_step == 7
    assert logps is None and extra == {}
    trace = [json.loads(s) for s in path.read_text().splitlines()]
    assert sum(t["active_sequences"] for t in trace) == 12
    assert all(t["logical_batch_size"] == 12 for t in trace)


def test_unsupported_backend_and_payload_fail_before_misalignment():
    trainer = Chunked(generation_micro_batch_size=1)
    trainer.use_vllm = True
    with pytest.raises(ValueError, match="ordinary Transformers"):
        trainer._generate_single_turn([[1]], None, {})
    trainer.use_vllm = False
    with pytest.raises(ValueError, match="text-only"):
        trainer._generate_single_turn([[1]], ["image"], {})
    with pytest.raises(ValueError, match="positive integer"):
        Chunked(generation_micro_batch_size=0)


def test_b_keeps_baseline_contract_except_intended_rollout_changes(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["--config", str(PROJECT_ROOT / "configs/training/grpo_experiment_b.yaml"),
                              "--include-synthetic", "--output-dir", str(tmp_path)])
    cfg = load_config(args)
    baseline = PROJECT_ROOT / cfg["comparison"]["baseline_run"] / "run_manifest.json"
    if not baseline.exists():
        pytest.skip("Historical private-data snapshot is not present in this checkout")
    splits, summary = prepare(args, cfg, tmp_path)
    base = json.loads(baseline.read_text(encoding="utf-8"))["run_contract"]
    current = run_contract(cfg, splits)
    assert summary["train"]["samples"] == 34 and not splits["validation"]
    assert current["data_sha256"] == base["data_sha256"]
    assert current["model"] == base["model"] and current["lora"] == base["lora"]
    assert current["reward_weights"] == base["reward_weights"]
    differences = {k for k, v in current["algorithm"].items() if v != base["algorithm"][k]}
    assert differences == {"num_generations", "generation_batch_size", "gradient_accumulation_steps"}
    t = cfg["training"]
    assert t["gradient_accumulation_steps"] * t["per_device_train_batch_size"] / t["num_generations"] == 2
    assert cfg["rollout"]["micro_batch_size"] == 1


def test_b_smoke_still_uses_six_candidates_and_explicit_1024_cap():
    args = build_parser().parse_args([
        "--config", str(PROJECT_ROOT / "configs/training/grpo_experiment_b.yaml"),
        "--smoke-test", "--max-completion-length", "1024",
    ])
    cfg = load_config(args)
    t = cfg["training"]
    assert t["num_generations"] == t["generation_batch_size"] == 6
    assert t["gradient_accumulation_steps"] == 12
    assert t["max_completion_length"] == 1024 and t["warmup_steps"] == 0


def test_plot_reader_deduplicates_resumes_and_ignores_partial_last_line(tmp_path: Path):
    (tmp_path / "progress.jsonl").write_text(
        '{"step":1,"reward":0.1}\n{"step":1,"reward":0.2}\n'
        '{"step":2,"train_runtime":1}\n{"step":', encoding="utf-8")
    assert reward_rows(tmp_path) == [{"step": 1, "reward": 0.2}]
