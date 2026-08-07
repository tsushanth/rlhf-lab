"""Shared config and dataset helpers for the RLHF lab."""
import argparse
import os
from datasets import load_dataset

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Anchor to the repo root regardless of cwd, so `results/` never ends up
# nested under scripts/ depending on where a script is invoked from.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def base_arg_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--max-samples", type=int, default=None,
                    help="cap dataset size for quick smoke tests")
    return p


def load_hh_rlhf(split="train", max_samples=None):
    """Anthropic/hh-rlhf: each row has `chosen` and `rejected` full transcripts
    (multi-turn, ending in '\\n\\nAssistant: ...'). We use the 'helpful-base'
    subset — cleanest single-turn-ish pairs for a first pass."""
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def split_prompt_response(transcript):
    """hh-rlhf packs prompt+response into one string ending in the final
    'Assistant:' turn. Split on the last occurrence to recover (prompt, response)."""
    marker = "\n\nAssistant:"
    idx = transcript.rfind(marker)
    prompt = transcript[:idx + len(marker)]
    response = transcript[idx + len(marker):].strip()
    return prompt, response
