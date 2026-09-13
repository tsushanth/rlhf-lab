"""Compute cheap automated metrics for any set of checkpoints.

No LLM judge calls — just model inference. Produces a JSON + markdown report
with perplexity, generation length, diversity, and reward-model scores. Use this
before (or alongside) the expensive judged eval in 05_eval.py.
"""
import os
import json
import math
import re
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from common import base_arg_parser, load_hh_rlhf, split_prompt_response, RESULTS_DIR


def parse_checkpoint_arg(s: str):
    """Parse 'name=path' or just 'path' (derive name from basename)."""
    if "=" in s:
        name, path = s.split("=", 1)
        return name.strip(), path.strip()
    path = s.strip()
    return os.path.basename(path.rstrip("/")), path


def discover_checkpoints(results_dir: str, include_base: bool = False):
    """Auto-discover causal-LM checkpoints under results/."""
    ckpts = {}
    if not os.path.isdir(results_dir):
        return ckpts
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if name.startswith("reward_model"):
            continue
        if os.path.isfile(os.path.join(path, "config.json")):
            ckpts[name] = path
    return ckpts


def load_models(checkpoints: dict, device: str):
    """Load all checkpoints into memory. Returns {name: (model, tokenizer)}."""
    models = {}
    for name, path in checkpoints.items():
        print(f"Loading {name} from {path} ...")
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        mdl.config.pad_token_id = tok.pad_token_id
        mdl.to(device).eval()
        models[name] = (mdl, tok)
    return models


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, texts, device, batch_size=4, max_length=1024):
    """Average perplexity over a list of strings. Lower is better."""
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            # labels = input_ids so loss is computed on all tokens
            out = model(**enc)
            # Cross-entropy loss per token = total_nll / total_tokens in the batch
            # HF returns mean over batch; recover total from number of non-pad tokens
            loss = out.loss.item()
            # More precise: compute token-level average
            # out.logits shape: [batch, seq_len, vocab]
            logits = out.logits
            labels = enc.input_ids
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            active = shift_labels != tokenizer.pad_token_id
            active_labels = torch.where(active, shift_labels, torch.tensor(-100, device=device))
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), active_labels.view(-1))
            n_tokens = active.sum().item()
            total_nll += nll.item()
            total_tokens += n_tokens
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def generate_responses(model, tokenizer, prompts, device, max_new_tokens=128, batch_size=4):
    """Generate one response per prompt."""
    responses = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
            for j, seq in enumerate(out):
                # Decode only the newly generated tokens
                prompt_len = enc.input_ids[j].ne(tokenizer.pad_token_id).sum().item()
                gen_ids = seq[prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                responses.append(text)
    return responses


def average_length(responses):
    """Average character length."""
    if not responses:
        return 0.0
    return sum(len(r) for r in responses) / len(responses)


def distinct_n(responses, n=1):
    """Distinct-n diversity ratio: unique n-grams / total n-grams."""
    total_grams = 0
    unique_grams = set()
    for r in responses:
        tokens = r.split()
        if len(tokens) < n:
            continue
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total_grams += len(grams)
        unique_grams.update(grams)
    if total_grams == 0:
        return 0.0
    return len(unique_grams) / total_grams


# ---------------------------------------------------------------------------
# Reward model scoring
# ---------------------------------------------------------------------------

def score_with_reward_model(reward_model_path, tokenizer, checkpoints_models, preference_pairs, device):
    """Score (prompt+chosen) and (prompt+rejected) with a reward model.

    Returns dict of {checkpoint_name: {"chosen_mean": float, "rejected_mean": float,
                                      "margin_mean": float}}
    """
    from transformers import AutoModelForSequenceClassification
    rm_tok = AutoTokenizer.from_pretrained(reward_model_path)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    rm = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1, torch_dtype=torch.bfloat16)
    rm.config.pad_token_id = rm_tok.pad_token_id
    rm.to(device).eval()

    # Build transcripts in the RM tokenizer's vocabulary
    chosen_texts = [ex["chosen"] for ex in preference_pairs]
    rejected_texts = [ex["rejected"] for ex in preference_pairs]

    def score_batch(texts):
        with torch.no_grad():
            enc = rm_tok(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024).to(device)
            return rm(**enc).logits.squeeze(-1).cpu().float().tolist()

    chosen_scores = []
    rejected_scores = []
    batch_size = 8
    for i in range(0, len(chosen_texts), batch_size):
        chosen_scores.extend(score_batch(chosen_texts[i:i + batch_size]))
        rejected_scores.extend(score_batch(rejected_texts[i:i + batch_size]))

    # Per-checkpoint: score the *generated* responses too if we had them,
    # but for now just report the RM's raw behavior on the human pairs.
    # A well-trained policy should produce responses that score closer to chosen.
    margins = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    overall = {
        "chosen_mean": sum(chosen_scores) / len(chosen_scores),
        "rejected_mean": sum(rejected_scores) / len(rejected_scores),
        "margin_mean": sum(margins) / len(margins),
        "margin_std": (sum((m - sum(margins)/len(margins))**2 for m in margins) / len(margins))**0.5,
    }
    rm.cpu()
    del rm
    torch.cuda.empty_cache()
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = base_arg_parser("Automated metrics for trained checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Explicit checkpoints as name=path pairs. If omitted, auto-discovers from results/")
    parser.add_argument("--include-base", action="store_true",
                        help="Also evaluate the raw base model (Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--reward-model", default=None,
                        help="Path to a trained reward model for scoring generations")
    parser.add_argument("--n-prompts", type=int, default=100,
                        help="Number of held-out prompts to generate from")
    parser.add_argument("--perplexity-samples", type=int, default=200,
                        help="Number of held-out texts for perplexity")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Resolve checkpoints
    if args.checkpoints:
        checkpoints = dict(parse_checkpoint_arg(c) for c in args.checkpoints)
    else:
        checkpoints = discover_checkpoints(args.results_dir)
    if args.include_base:
        from common import BASE_MODEL
        checkpoints["base"] = BASE_MODEL
    if len(checkpoints) < 1:
        raise SystemExit("No checkpoints found. Train something first or pass --checkpoints.")

    print(f"Evaluating {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")

    device = args.device
    models = load_models(checkpoints, device)

    # Load held-out data
    raw_test = load_hh_rlhf(split="test", max_samples=max(args.n_prompts, args.perplexity_samples))
    prompts = [split_prompt_response(ex["chosen"])[0] for ex in raw_test.select(range(args.n_prompts))]
    perplexity_texts = [ex["chosen"] for ex in raw_test.select(range(min(args.perplexity_samples, len(raw_test))))]

    # Reward model scoring (once, on the human pairs — not per-checkpoint)
    rm_summary = None
    if args.reward_model and os.path.exists(args.reward_model):
        print(f"\nScoring held-out pairs with reward model: {args.reward_model}")
        # Use a small subset for RM scoring to keep it fast
        rm_pairs = raw_test.select(range(min(100, len(raw_test))))
        rm_summary = score_with_reward_model(args.reward_model, None, models,
                                             [rm_pairs[i] for i in range(len(rm_pairs))], device)
        print(f"  RM chosen_mean={rm_summary['chosen_mean']:.3f} rejected_mean={rm_summary['rejected_mean']:.3f} "
              f"margin={rm_summary['margin_mean']:.3f} (std={rm_summary['margin_std']:.3f})")

    results = {}
    print("\n--- Generating responses & computing metrics ---")
    for name, (mdl, tok) in models.items():
        print(f"\n{name}:")
        # 1. Perplexity on held-out chosen text
        ppl = compute_perplexity(mdl, tok, perplexity_texts, device, batch_size=args.batch_size)
        print(f"  perplexity = {ppl:.2f}")

        # 2. Generate responses
        responses = generate_responses(mdl, tok, prompts, device, batch_size=args.batch_size)

        # 3. Length
        avg_len = average_length(responses)
        print(f"  avg_length = {avg_len:.1f} chars")

        # 4. Diversity
        d1 = distinct_n(responses, n=1)
        d2 = distinct_n(responses, n=2)
        print(f"  distinct-1 = {d1:.3f}  distinct-2 = {d2:.3f}")

        # 5. Optional: score generations with reward model
        rm_gen_scores = None
        if args.reward_model and os.path.exists(args.reward_model):
            from transformers import AutoModelForSequenceClassification
            rm_tok = AutoTokenizer.from_pretrained(args.reward_model)
            if rm_tok.pad_token is None:
                rm_tok.pad_token = rm_tok.eos_token
            rm = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, torch_dtype=torch.bfloat16)
            rm.config.pad_token_id = rm_tok.pad_token_id
            rm.to(device).eval()
            scores = []
            with torch.no_grad():
                for i in range(0, len(responses), args.batch_size):
                    batch = responses[i:i + args.batch_size]
                    enc = rm_tok(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=1024).to(device)
                    s = rm(**enc).logits.squeeze(-1).cpu().float().tolist()
                    scores.extend(s if isinstance(s, list) else [s])
            rm.cpu()
            del rm
            torch.cuda.empty_cache()
            rm_gen_scores = {"mean": sum(scores) / len(scores), "std": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5}
            print(f"  rm_score   = {rm_gen_scores['mean']:.3f} (std={rm_gen_scores['std']:.3f})")

        results[name] = {
            "perplexity": ppl,
            "avg_length": avg_len,
            "distinct_1": d1,
            "distinct_2": d2,
            "rm_gen_scores": rm_gen_scores,
        }

    # If RM summary exists, attach it to every result for reference
    if rm_summary:
        for r in results.values():
            r["rm_reference"] = rm_summary

    # Save JSON
    out_json = os.path.join(args.results_dir, "automated_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {out_json}")

    # Save markdown report
    out_md = os.path.join(args.results_dir, "automated_metrics.md")
    with open(out_md, "w") as f:
        f.write("# Automated Metrics Report\n\n")
        f.write(f"Held-out prompts: {args.n_prompts} | Perplexity samples: {args.perplexity_samples}\n\n")
        if rm_summary:
            f.write(f"Reward model reference (human pairs): chosen={rm_summary['chosen_mean']:.3f} "
                    f"rejected={rm_summary['rejected_mean']:.3f} margin={rm_summary['margin_mean']:.3f}\n\n")
        f.write("| Checkpoint | Perplexity ↓ | Avg Length | Distinct-1 ↑ | Distinct-2 ↑ | RM Score\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, r in results.items():
            rm = r["rm_gen_scores"]
            rm_str = f"{rm['mean']:.3f}±{rm['std']:.3f}" if rm else "n/a"
            f.write(f"| {name} | {r['perplexity']:.2f} | {r['avg_length']:.1f} | {r['distinct_1']:.3f} | {r['distinct_2']:.3f} | {rm_str}\n")
        f.write("\n**Legend:**\n")
        f.write("- **Perplexity** = exp(avg NLL) on held-out human chosen responses. Lower = better at predicting real text.\n")
        f.write("- **Distinct-n** = unique n-grams / total n-grams in generated responses. Higher = more diverse.\n")
        f.write("- **RM Score** = reward model mean on generated responses. Higher = more preferred by RM.\n")
        if rm_summary:
            f.write(f"- Compare RM Score to RM reference: if a checkpoint scores near {rm_summary['rejected_mean']:.2f} it's RM-rejected level; "
                    f"near {rm_summary['chosen_mean']:.2f} is RM-chosen level.\n")
    print(f"Report saved: {out_md}")


if __name__ == "__main__":
    main()
