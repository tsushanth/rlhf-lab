"""Label candidate pairs with an AI judge (Claude) instead of humans.

This is the actual RLIAF step: same pairwise-preference schema as HH-RLHF's
human labels, but the label comes from an LLM judge on a fixed rubric. Output
is written in the same '<prompt>\\n\\nAssistant: <response>' transcript
format HH-RLHF uses, so 02_reward_model.py / 04_dpo.py consume it via
--data-file with zero code changes -- the labeler is swapped, nothing else.
"""
import os
import json
import random
from common import base_arg_parser, RESULTS_DIR

JUDGE_PROMPT = """You are labeling which of two AI assistant responses is more helpful,
honest, and harmless, for training data purposes. Pick the better one.

Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Reply with exactly one token: "A" or "B". No ties -- if genuinely equal, pick
either, but do not reply anything else.
"""


def judge(client, prompt, response_a, response_b, max_retries=2):
    text = JUDGE_PROMPT.format(prompt=prompt, response_a=response_a, response_b=response_b)
    for attempt in range(max_retries + 1):
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=20,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": text}],
        )
        text_block = next((b for b in resp.content if b.type == "text"), None)
        if text_block is not None:
            verdict = text_block.text.strip().upper()
            return "A" if "A" in verdict and "B" not in verdict else "B"
        print(f"WARN: judge response had no text block (attempt {attempt + 1}/{max_retries + 1}), stop_reason={resp.stop_reason}", flush=True)
    print("WARN: judge never returned a text block after retries, defaulting to 'A' for this pair", flush=True)
    return "A"


def main():
    parser = base_arg_parser("Label RLAIF candidate pairs with a Claude judge")
    parser.add_argument("--candidates-file", default=os.path.join(RESULTS_DIR, "rlaif_candidates.jsonl"))
    parser.add_argument("--out-file", default=os.path.join(RESULTS_DIR, "rlaif_preference_pairs.jsonl"))
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    candidates = []
    with open(args.candidates_file) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    if args.max_samples:
        candidates = candidates[: args.max_samples]

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as out:
        for i, ex in enumerate(candidates):
            # De-bias position preference: swap which side is "A" for half
            # the judge calls, then swap the verdict back.
            swap = random.random() < 0.5
            a, b = (ex["response_b"], ex["response_a"]) if swap else (ex["response_a"], ex["response_b"])
            verdict = judge(client, ex["prompt"], a, b)
            if swap:
                verdict = "B" if verdict == "A" else "A"

            chosen = ex["response_a"] if verdict == "A" else ex["response_b"]
            rejected = ex["response_b"] if verdict == "A" else ex["response_a"]

            out.write(json.dumps({
                "chosen": f"{ex['prompt']} {chosen}",
                "rejected": f"{ex['prompt']} {rejected}",
            }) + "\n")

            if i % 50 == 0:
                print(f"{i}/{len(candidates)} pairs judged", flush=True)

    print(f"Wrote {len(candidates)} AI-labeled preference pairs to {args.out_file}")


if __name__ == "__main__":
    main()
