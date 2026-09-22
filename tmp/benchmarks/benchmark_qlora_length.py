from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.train()

    from bitsandbytes.optim import PagedAdamW8bit

    optimizer = PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4
    )
    input_ids = torch.randint(
        low=100,
        high=model.config.vocab_size - 1,
        size=(1, args.length),
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, : min(64, args.length // 4)] = -100

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "length": args.length,
                    "success": False,
                    "error": "CUDA out of memory",
                    "detail": str(exc),
                    "peak_allocated_mib": round(
                        torch.cuda.max_memory_allocated() / 1024**2, 2
                    ),
                    "peak_reserved_mib": round(
                        torch.cuda.max_memory_reserved() / 1024**2, 2
                    ),
                    "free_before_mib": round(free_before / 1024**2, 2),
                    "total_mib": round(total / 1024**2, 2),
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2) from exc

    elapsed = time.perf_counter() - started
    free_after, _ = torch.cuda.mem_get_info()
    print(
        json.dumps(
            {
                "length": args.length,
                "success": True,
                "batch_size": 1,
                "dtype": str(dtype),
                "loss": float(output.loss.detach()),
                "seconds": round(elapsed, 3),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
                "free_before_mib": round(free_before / 1024**2, 2),
                "free_after_mib": round(free_after / 1024**2, 2),
                "total_mib": round(total / 1024**2, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
