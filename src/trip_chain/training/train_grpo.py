"""Direct Qwen GRPO from fresh LoRA parameters; never calls SFTTrainer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import warnings
from pathlib import Path
from typing import Any

import yaml

from trip_chain.data.grpo import (
    SYSTEM_PROMPT,
    load_annotations,
    make_training_row,
    split_records,
    target_without_evidence,
    write_json,
    write_splits,
)
from trip_chain.training.grpo_rewards import REWARD_NAMES, build_reward_functions

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct GRPO on GLiNER2 labels, without SFT")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/training/grpo_6gb.yaml")
    p.add_argument("--annotations", type=Path, nargs="+", help="Override input GLiNER2 JSONL files")
    p.add_argument("--include-synthetic", action="store_true", help="Explicitly include synthetic data")
    p.add_argument("--model-name-or-path", help="Base model; no SFT adapter is loaded")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--prepare-only", action="store_true", help="CPU-only data validation and splitting")
    p.add_argument("--dry-run", action="store_true", help="Also check tokenizer budgets; no model/training")
    p.add_argument("--smoke-test", action="store_true", help="One optimizer step on small real samples")
    p.add_argument("--resume-from-checkpoint", type=Path, help="Resume this GRPO run, not SFT")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-prompt-length", type=int)
    p.add_argument("--max-completion-length", type=int)
    return p


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    for name in ("max_prompt_length", "max_completion_length"):
        value = getattr(args, name)
        if value is not None:
            config["training"][name] = value
    if args.model_name_or_path:
        config["model"]["model_name_or_path"] = args.model_name_or_path
    if args.output_dir:
        config["training"]["output_dir"] = str(project_path(args.output_dir))
    if args.smoke_test:
        group = config["training"]["num_generations"]
        micro = config.get("rollout", {}).get("micro_batch_size")
        config["training"].update(
            max_steps=1, gradient_accumulation_steps=2 * group if micro else group,
            generation_batch_size=group, save_strategy="no", eval_strategy="no",
            warmup_steps=0,
            max_completion_length=(args.max_completion_length if args.max_completion_length
                                   is not None else
                                   min(512, config["training"]["max_completion_length"])),
        )
        if not args.output_dir:
            config["training"]["output_dir"] += "-smoke"
    weights = config["reward_weights"]
    if set(weights) != set(REWARD_NAMES):
        raise ValueError(f"reward_weights must contain exactly {REWARD_NAMES}")
    if any(not isinstance(w, (int, float)) or not 0 <= w < float("inf")
           for w in weights.values()) or not any(weights.values()):
        raise ValueError("Reward weights must be finite, nonnegative, and not all zero")
    if config["training"]["num_generations"] < 2:
        raise ValueError("GRPO needs at least two completions per prompt")
    if config["training"]["beta"] < 0:
        raise ValueError("beta cannot be negative")
    micro = config.get("rollout", {}).get("micro_batch_size")
    if micro is not None:
        if type(micro) is not int or micro < 1:
            raise ValueError("rollout.micro_batch_size must be a positive integer")
        if config["training"].get("use_vllm") or config["training"].get("use_transformers_paged"):
            raise ValueError("Micro-batch rollouts require ordinary Transformers generation")
        if micro > config["training"]["generation_batch_size"]:
            raise ValueError("Rollout micro batch cannot exceed the logical generation batch")
    return config


def prepare(args: argparse.Namespace, config: dict[str, Any], run_dir: Path) -> tuple[dict, dict]:
    data = config["data"]
    paths = [project_path(p) for p in (args.annotations or data["annotation_files"])]
    if args.include_synthetic and data.get("synthetic_file"):
        synthetic_path = project_path(data["synthetic_file"])
        if synthetic_path not in paths:
            paths.append(synthetic_path)
    records = load_annotations(paths, allow_synthetic=args.include_synthetic)
    splits = split_records(records, data["validation_ratio"], config["training"]["seed"])
    if not any(c["nodes"] for r in splits["train"] for c in r["chains"]):
        raise ValueError("Training split has no activity nodes; adjust data or validation_ratio")
    if args.resume_from_checkpoint:
        manifest_path = project_path(args.resume_from_checkpoint).parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Resume requires this script's GRPO run_manifest.json beside checkpoints")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_contract") != run_contract(config, splits):
            raise ValueError("Resume data/model/reward/LoRA settings differ from the original GRPO run")
    summary = write_splits(splits, run_dir / "data")
    write_json(run_dir / "resolved_config.json", config)
    return splits, summary


def run_contract(config: dict, splits: dict) -> dict:
    data = {name: [{"id": r["chunk_id"], "text_sha256": r["text_sha256"],
                    "chains": r["chains"]} for r in records]
            for name, records in splits.items()}
    signature = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    training = config["training"]
    contract = {
        "mode": "direct_grpo_no_sft", "model": config["model"], "lora": config["lora"],
        "reward_weights": config["reward_weights"], "system_prompt": SYSTEM_PROMPT,
        "data_sha256": hashlib.sha256(signature).hexdigest(),
        "algorithm": {k: training[k] for k in (
            "beta", "loss_type", "scale_rewards", "multi_objective_aggregation", "seed",
            "num_generations", "generation_batch_size", "gradient_accumulation_steps",
            "num_iterations", "epsilon", "max_completion_length", "temperature", "top_p",
            "learning_rate", "warmup_steps", "lr_scheduler_type", "optim", "max_grad_norm",
            "per_device_train_batch_size", "mask_truncated_completions",
            "use_bias_correction_kl", "max_prompt_length", "generation_kwargs", "top_k",
            "importance_sampling_level",
        )},
    }
    # Preserve resume compatibility for historical runs without this option.
    if config.get("rollout"):
        contract["rollout"] = config["rollout"]
    return contract


def tokenize_splits(splits: dict, tokenizer: Any, training: dict) -> tuple[dict, dict]:
    result, lengths = {}, {}
    max_prompt = training["max_prompt_length"]
    max_output = training["max_completion_length"]
    if max_prompt < 1 or max_output < 1:
        raise ValueError("Token budgets must be positive")
    for split, records in splits.items():
        rows, sizes = [], []
        for record in records:
            row = make_training_row(record, tokenizer)
            prompt_len = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            target = json.dumps(target_without_evidence(record["chains"]),
                                ensure_ascii=False, separators=(",", ":"))
            target_len = len(tokenizer(target, add_special_tokens=False)["input_ids"]) + 1
            sizes.append({"sample_id": row["sample_id"], "prompt": prompt_len,
                          "gold_completion": target_len})
            if prompt_len > max_prompt or target_len > max_output:
                raise ValueError(
                    f"{row['sample_id']}: prompt={prompt_len}/{max_prompt}, "
                    f"gold_completion={target_len}/{max_output} tokens. "
                    "Increase the budgets or supply a shorter, fully re-annotated passage. "
                    "Text and labels will never be silently truncated."
                )
            rows.append(row)
        result[split], lengths[split] = rows, sizes
    return result, lengths


def trainer_kwargs(training: dict, weights: dict, *, bf16: bool) -> dict[str, Any]:
    # max_prompt_length was removed from TRL 0.29.1: enforce it ourselves, before training.
    values = {k: v for k, v in training.items() if k != "max_prompt_length"}
    values.update(
        output_dir=str(project_path(training["output_dir"])),
        reward_weights=[weights[name] for name in REWARD_NAMES],
        bf16=bf16, fp16=not bf16, report_to="none", push_to_hub=False,
        remove_unused_columns=False, disable_dropout=True,
    )
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This 6GB preset supports one GPU/process; use a single-process launch")
    config = load_config(args)
    training = config["training"]
    output_dir = project_path(training["output_dir"])
    # Refuse accidental overwrite of an existing trained adapter/checkpoint.
    if not (args.prepare_only or args.dry_run or args.resume_from_checkpoint):
        occupied = output_dir.exists() and (
            (output_dir / "adapter_config.json").exists()
            or any(output_dir.glob("checkpoint-*"))
        )
        if occupied:
            raise ValueError("Output already has a model; set --output-dir or resume its checkpoint")
    if args.resume_from_checkpoint:
        resume = project_path(args.resume_from_checkpoint)
        if not (resume / "trainer_state.json").is_file():
            raise ValueError("Resume requires a full GRPO checkpoint with trainer_state.json")
    preparation_dir = output_dir
    if (args.prepare_only or args.dry_run) and (output_dir / "run_manifest.json").exists():
        preparation_dir = output_dir / "preflight"
    splits, summary = prepare(args, config, preparation_dir)
    if args.prepare_only:
        return {"mode": "prepare_only", "output_dir": str(preparation_dir), "data": summary}

    from transformers import AutoConfig, AutoTokenizer

    model_name = config["model"]["model_name_or_path"]
    if (Path(model_name) / "adapter_config.json").is_file():
        raise ValueError("Supply base model weights, not an SFT/LoRA adapter directory")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("A tokenizer with an EOS token is required")
    if args.smoke_test:
        # Use complete real passages and labels; never crop a positive example into a negative.
        def size(r: dict) -> int:
            return len(tokenizer(json.dumps(target_without_evidence(r["chains"])),
                                 add_special_tokens=False)["input_ids"])

        positive = [r for r in splits["train"] if any(c["nodes"] for c in r["chains"])]
        negative = [r for r in splits["train"] if not r["chains"]]
        splits = {"train": [min(positive, key=size)], "validation": []}
        if negative:
            splits["train"].append(min(negative, key=lambda r: len(r["text"])))
        summary = write_splits(splits, preparation_dir / "data")
    rows, lengths = tokenize_splits(splits, tokenizer, training)
    write_json(preparation_dir / "data/token_lengths.json", lengths)
    model_info = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    context_limit = getattr(model_info, "max_position_embeddings", None)
    largest_prompt = max(x["prompt"] for sizes in lengths.values() for x in sizes)
    if context_limit and largest_prompt + training["max_completion_length"] > context_limit:
        raise ValueError("Prompt plus completion budget exceeds model context length")
    if args.dry_run:
        return {"mode": "dry_run", "data": summary, "token_lengths": lengths}
    if training.get("eval_strategy") != "no" and not rows["validation"]:
        raise ValueError("Evaluation is enabled but the validation split is empty")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if importlib.metadata.version("trl") != "0.29.1":
        raise RuntimeError("This script targets trl==0.29.1; install requirements/grpo.txt")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --prepare-only/--dry-run for CPU validation")
    set_seed(training["seed"])
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg = GRPOConfig(**trainer_kwargs(training, config["reward_weights"],
                                     bf16=compute_dtype == torch.bfloat16))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map={"": 0}, dtype=compute_dtype,
        attn_implementation="sdpa", trust_remote_code=False, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(**config["lora"])

    class RewardHealth(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None,
                   **kwargs: Any) -> None:
            if logs and state.is_world_process_zero:
                with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": state.global_step,
                                             "epoch": state.epoch, **logs}) + "\n")
            if logs and logs.get("frac_reward_zero_std", 0) >= 0.95:
                warnings.warn(
                    "Almost all candidate groups have identical rewards: the policy has "
                    "little task-learning signal. Inspect reward_trace.jsonl and JSON validity.",
                    stacklevel=2,
                )

    trace_path = output_dir / "reward_trace.jsonl" if config.get("save_reward_trace", True) else None
    trainer_type, extra_kwargs = GRPOTrainer, {}
    micro = config.get("rollout", {}).get("micro_batch_size")
    if micro is not None:
        from trip_chain.training.grpo_microbatch import MicroBatchGenerationMixin

        class SerialGRPOTrainer(MicroBatchGenerationMixin, GRPOTrainer):
            pass

        trainer_type = SerialGRPOTrainer
        extra_kwargs = {"generation_micro_batch_size": micro,
                        "generation_trace_path": output_dir / "rollout_microbatches.jsonl"}
    trainer = trainer_type(
        model=model, args=cfg, processing_class=tokenizer, peft_config=lora,
        train_dataset=Dataset.from_list(rows["train"]),
        eval_dataset=Dataset.from_list(rows["validation"]) if rows["validation"] else None,
        reward_funcs=build_reward_functions(trace_path), callbacks=[RewardHealth()],
        **extra_kwargs,
    )
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "trl", "peft", "bitsandbytes")}
    write_json(output_dir / "run_manifest.json", {
        "training_mode": "direct_grpo_no_sft", "model": model_name, "versions": versions,
        "reference_policy": "frozen base model via TRL's disabled-adapter reference",
        "kl": "TRL sampled k3 estimator: exp(logp_ref-logp)- (logp_ref-logp) - 1",
        "beta": cfg.beta, "reward_order": REWARD_NAMES, "data": summary,
        "run_contract": run_contract(config, splits),
        "smoke_test": args.smoke_test,
        "rollout": {"logical_batch_size": cfg.generation_batch_size,
                    "micro_batch_size": micro, "full_group_scoring_by_trl": True},
    })
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(
        resume_from_checkpoint=str(project_path(args.resume_from_checkpoint))
        if args.resume_from_checkpoint else None
    )
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    metrics = dict(result.metrics)
    metrics.update(gpu=torch.cuda.get_device_name(0),
                   peak_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                   smoke_test=args.smoke_test)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if rows["validation"] and training.get("eval_strategy") != "no":
        trainer.save_metrics("eval", trainer.evaluate())
    return {"output_dir": str(output_dir), "metrics": metrics}


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
