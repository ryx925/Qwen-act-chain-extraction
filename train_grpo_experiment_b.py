"""Six-candidate serial GRPO, with automatic reward plots against historical A."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trip_chain.training.train_grpo import build_parser, load_config, project_path, run


def main():
    parser = build_parser()
    parser.description = "Experiment B: six candidates per passage, memory-bounded generation"
    parser.set_defaults(config=ROOT / "configs/training/grpo_experiment_b.yaml",
                        include_synthetic=True)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        if args.resume_from_checkpoint:
            args.output_dir = project_path(args.resume_from_checkpoint).parent
        else:
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
            mode = "smoke" if args.smoke_test else "full"
            args.output_dir = ROOT / f"outputs/experiments/grpo_b_g6_serial_{mode}_{stamp}"
    config = load_config(args)
    if config["training"]["num_generations"] != 6:
        parser.error("Experiment B must keep num_generations=6")
    if not config.get("rollout", {}).get("micro_batch_size"):
        parser.error("Experiment B requires rollout.micro_batch_size")
    result = run(args)
    if not (args.prepare_only or args.dry_run or args.skip_plot):
        from trip_chain.training.grpo_plots import plot_comparison

        result["reward_comparison"] = plot_comparison(
            project_path(config["comparison"]["baseline_run"]), project_path(args.output_dir)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
