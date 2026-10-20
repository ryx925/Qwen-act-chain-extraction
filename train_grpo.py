"""Run from any directory: python /path/to/Qwen/train_grpo.py [options]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trip_chain.training.train_grpo import main

if __name__ == "__main__":
    main()
