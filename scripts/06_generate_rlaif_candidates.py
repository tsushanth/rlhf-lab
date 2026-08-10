"""Sample two candidate completions per prompt from the SFT checkpoint.

RLAIF replaces the human labeler with an AI judge, but the prompts and
candidate-generation step are unchanged from classic RLHF data collection.
We deliberately reuse the *same* HH-RLHF prompts that were human-labeled,
rather than a fresh prompt set -- that makes the eventual human-vs-AI-label
comparison an apples-to-apples ablation instead of also varying the prompts.

Output: one JSON line per prompt with {prompt, response_a, response_b}.
07_judge_rlaif_pairs.py consumes this and produces the labeled pairs.
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import base_arg_parser, load_hh_rlhf, split_prompt_response, RESULTS_DIR


def main():
    parser = base_arg_parser("Generate candidate completion pairs for RLAIF labeling")
    parser.add_argument("--sft-checkpoint", default=os.path.join(RESULTS_DIR, "sft"))
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--out-file", default=os.path.join(RESULTS_DIR, "rlaif_candidates.jsonl"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    raw = load_hh_rlhf(max_samples=args.n_prompts)
    prompts = [split_prompt_response(ex["chosen"])[0] for ex in raw]

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            # Two samples at different temperatures so the pair is genuinely
            # distinguishable -- same temp twice mostly produces near-ties,
            # which wastes judge calls on pairs with no real signal.
            out_a = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                    temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            out_b = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                    temperature=1.2, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
            resp_a = tokenizer.decode(out_a[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            resp_b = tokenizer.decode(out_b[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            f.write(json.dumps({"prompt": prompt, "response_a": resp_a, "response_b": resp_b}) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(prompts)} candidate pairs generated", flush=True)

    print(f"Wrote {len(prompts)} candidate pairs to {args.out_file}")


if __name__ == "__main__":
    main()
