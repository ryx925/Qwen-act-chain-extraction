"""Chunk text rollouts without changing TRL's full-group scoring or loss."""
from __future__ import annotations

import json
import time
from pathlib import Path


class MicroBatchGenerationMixin:
    """Only override the ordinary Transformers generation path in TRL 0.29.1.

    TRL has already repeated prompts G times before this method. Return the same
    number of candidates in the same order; TRL then scores/normalizes the full
    group. Never reseed, filter candidates, or update parameters between chunks.
    """

    def __init__(self, *args, generation_micro_batch_size: int,
                 generation_trace_path: Path | None = None, **kwargs):
        if type(generation_micro_batch_size) is not int or generation_micro_batch_size < 1:
            raise ValueError("generation_micro_batch_size must be a positive integer")
        self.generation_micro_batch_size = generation_micro_batch_size
        self.generation_trace_path = generation_trace_path
        self._microbatch_call = 0
        super().__init__(*args, **kwargs)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        if images is not None or multimodal_fields:
            raise ValueError("Micro-batch generation supports text-only inputs")
        if self.use_vllm or self.use_transformers_paged:
            raise ValueError("Micro-batch generation requires ordinary Transformers generation")
        completions = []
        step = self.state.global_step
        self._microbatch_call += 1
        for start in range(0, len(prompt_ids), self.generation_micro_batch_size):
            stop = min(start + self.generation_micro_batch_size, len(prompt_ids))
            started = time.perf_counter()
            outputs, logprobs, extra = super()._generate_single_turn(
                prompt_ids[start:stop], None, {}
            )
            if len(outputs) != stop - start:
                raise RuntimeError("Generation lost candidates; refusing to misalign rewards")
            if logprobs is not None or extra:
                raise RuntimeError("Unexpected generation payload for the pinned Transformers path")
            if self.state.global_step != step:
                raise RuntimeError("Policy updated while collecting a candidate group")
            completions.extend(outputs)
            if self.generation_trace_path is not None:
                self.generation_trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.generation_trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "step": step, "call": self._microbatch_call,
                        "start": start, "stop": stop, "active_sequences": stop - start,
                        "logical_batch_size": len(prompt_ids),
                        "seconds": round(time.perf_counter() - started, 4),
                        "completion_lengths": [len(ids) for ids in outputs],
                    }) + "\n")
        return completions, None, {}
