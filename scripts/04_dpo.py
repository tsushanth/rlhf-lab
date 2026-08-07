"""DPO: optimize directly on preference pairs, no reward model, no rollouts.

Same SFT starting checkpoint and same data as the PPO path in 03_ppo.py —
the point of this repo is that this script is dramatically simpler (one
model in memory instead of four, no sampling loop, no reward hacking to
guard against) and, per the DPO paper and most practitioner reports,
matches or beats PPO on preference win-rate for a fraction of the
engineering effort. 05_eval.py is where you check whether that holds here.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from common import base_arg_parser, load_hh_rlhf, split_prompt_response, RESULTS_DIR


def main():
    parser = base_arg_parser("DPO fine-tune directly on preference pairs")
    parser.add_argument("--sft-checkpoint", default=os.path.join(RESULTS_DIR, "sft"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1,
                         help="how strongly to penalize deviating from the SFT reference policy")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "dpo")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        # Qwen2.5 has no BOS token, but trl 0.9.x's DPOTrainer.tokenize_row
        # unconditionally prepends bos_token_id -> crashes on None. Alias it
        # to eos_token_id, which is a harmless no-op for this model family.
        tokenizer.bos_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    raw = load_hh_rlhf(max_samples=args.max_samples)

    # DPOTrainer wants separate prompt / chosen / rejected columns, where
    # chosen and rejected are just the completions. hh-rlhf packs the whole
    # transcript into each field, so split off the shared prompt prefix.
    def split_pair(ex):
        prompt, chosen_response = split_prompt_response(ex["chosen"])
        _, rejected_response = split_prompt_response(ex["rejected"])
        return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}

    train_ds = raw.map(split_pair, remove_columns=raw.column_names)

    config = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=args.beta,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_length=1024,
        max_prompt_length=512,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"DPO checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
