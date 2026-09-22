from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 with 4-bit QLoRA")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/qwen3_1_7b.yaml"))
    parser.add_argument(
        "--training-config", type=Path, default=Path("configs/training/qlora_6gb.yaml")
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--validation-file", type=Path, default=Path("data/processed/validation.jsonl")
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Override the configured model ID with a local model directory or another ID",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run one optimizer step")
    return parser


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")

    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)
    quantization = model_config["quantization"]
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"],
    )
    model_name = args.model_name_or_path or model_config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    data_files = {"train": str(args.train_file)}
    if args.validation_file.exists():
        data_files["validation"] = str(args.validation_file)
    dataset = load_dataset("json", data_files=data_files)

    lora = training_config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )

    output_dir = training_config["output_dir"]
    config_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        # Transformers 5 accepts a fractional warmup through warmup_steps.
        "warmup_steps": training_config["warmup_ratio"],
        "logging_steps": training_config["logging_steps"],
        "eval_strategy": training_config["eval_strategy"] if "validation" in dataset else "no",
        "eval_steps": training_config["eval_steps"],
        "save_strategy": training_config["save_strategy"],
        "save_steps": training_config["save_steps"],
        "save_total_limit": training_config["save_total_limit"],
        "max_length": training_config["max_length"],
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "tf32": training_config["tf32"],
        "optim": training_config["optim"],
        "seed": training_config["seed"],
        "completion_only_loss": True,
        "report_to": "none",
    }
    if args.smoke_test:
        output_dir = "outputs/checkpoints/smoke-test"
        config_kwargs.update(
            output_dir=output_dir,
            max_steps=1,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            max_length=min(256, training_config["max_length"]),
            eval_strategy="no",
            save_strategy="no",
            logging_steps=1,
        )
        dataset["train"] = dataset["train"].select(range(min(2, len(dataset["train"]))))

    sft_config = SFTConfig(**config_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = dict(result.metrics)
    metrics.update(
        gpu=torch.cuda.get_device_name(0),
        peak_memory_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        smoke_test=args.smoke_test,
    )
    metrics_path = Path("outputs/metrics") / (
        "smoke_train_metrics.json" if args.smoke_test else "train_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
