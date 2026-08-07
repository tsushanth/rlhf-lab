"""Warm-start SFT on the `chosen` responses from HH-RLHF.

Both PPO and DPO start from this checkpoint rather than the raw instruct
model — this isolates what PPO/DPO themselves contribute, since the base
Qwen2.5-Instruct model is already instruction-tuned on different data.
"""
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from common import base_arg_parser, load_hh_rlhf, split_prompt_response


def build_sft_dataset(raw):
    rows = []
    for ex in raw:
        prompt, response = split_prompt_response(ex["chosen"])
        rows.append({"text": f"{prompt} {response}"})
    return Dataset.from_list(rows)


def main():
    parser = base_arg_parser("SFT warm-start on HH-RLHF chosen responses")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "sft")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id

    raw = load_hh_rlhf(max_samples=args.max_samples)
    train_ds = build_sft_dataset(raw)

    config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"SFT checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
