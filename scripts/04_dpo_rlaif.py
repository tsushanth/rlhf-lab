"""Train DPO on the AI-judged (RLAIF) preference pairs.

Consumes the same 06/07 pipeline output as the human-labeled HH-RLHF data,
but swaps in Claude-judged pairs. After training, runs a quick head-to-head
against the human-labeled DPO checkpoint using automated metrics (fast) so
you get signal before paying for judged eval.
"""
import os
import sys
import subprocess
from common import base_arg_parser, RESULTS_DIR


def main():
    parser = base_arg_parser("DPO on RLAIF preference pairs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--data-file",
                        default=os.path.join(RESULTS_DIR, "rlaif_preference_pairs.jsonl"),
                        help="path to the RLAIF pairs produced by 07_judge_rlaif_pairs.py")
    parser.add_argument("--out-name", default="dpo_rlaif",
                        help="checkpoint subdir under results/")
    parser.add_argument("--compare-human-dpo", action="store_true",
                        help="after training, run automated metrics comparing dpo_rlaif vs dpo (human)")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        raise SystemExit(
            f"RLAIF data not found: {args.data_file}\n"
            f"Run first: python scripts/06_generate_rlaif_candidates.py\n"
            f"          python scripts/07_judge_rlaif_pairs.py"
        )

    # Delegate to the canonical DPO script — zero code duplication.
    dpo_cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "04_dpo.py"),
        "--data-file", args.data_file,
        "--out-name", args.out_name,
        "--epochs", str(args.epochs),
        "--beta", str(args.beta),
        "--results-dir", args.results_dir,
    ]
    if args.max_samples:
        dpo_cmd += ["--max-samples", str(args.max_samples)]

    print(f"Running: {' '.join(dpo_cmd)}")
    subprocess.run(dpo_cmd, check=True)

    # Quick automated comparison against human-labeled DPO (cheap, no API calls).
    if args.compare_human_dpo:
        human_dpo = os.path.join(args.results_dir, "dpo")
        rlaif_dpo = os.path.join(args.results_dir, args.out_name)
        if os.path.exists(human_dpo) and os.path.exists(rlaif_dpo):
            metrics_cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), "05b_automated_metrics.py"),
                "--checkpoints", f"dpo={human_dpo}", f"dpo_rlaif={rlaif_dpo}",
                "--results-dir", args.results_dir,
            ]
            print(f"\nRunning automated comparison: {' '.join(metrics_cmd)}")
            subprocess.run(metrics_cmd, check=True)
        else:
            print("[warn] Skipping comparison — one of the checkpoints is missing.")

    print(f"\nRLAIF DPO complete. Checkpoint: {os.path.join(args.results_dir, args.out_name)}")
    print("Next: include it in full eval with 05_eval_v2.py or 05_eval.py")


if __name__ == "__main__":
    main()
