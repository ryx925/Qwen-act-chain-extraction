"""Replot an in-progress or completed B run without loading a model."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trip_chain.training.grpo_plots import plot_comparison
from trip_chain.training.train_grpo import project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=Path(
        "outputs/experiments/grpo_all_annotations_20261019_200315/training"))
    args = parser.parse_args()
    print(json.dumps(plot_comparison(project_path(args.baseline_dir), project_path(args.run_dir)),
                     indent=2))
