"""Train a reward model on HH-RLHF preference pairs.

This is the piece classic RLHF needs that DPO skips entirely: a separate
model that scores (prompt, response) -> scalar, trained to prefer `chosen`
over `rejected`. PPO in 03_ppo.py optimizes the policy against this score.
"""
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from common import base_arg_parser, load_hh_rlhf


def main():
    parser = base_arg_parser("Train reward model on HH-RLHF pairs")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "reward_model")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    raw = load_hh_rlhf(max_samples=args.max_samples)

    # RewardTrainer (trl 0.9.x) expects pre-tokenized input_ids_chosen /
    # input_ids_rejected columns, not raw text.
    def tokenize_pair(ex):
        chosen = tokenizer(ex["chosen"], truncation=True, max_length=1024)
        rejected = tokenizer(ex["rejected"], truncation=True, max_length=1024)
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    train_ds = raw.map(tokenize_pair, remove_columns=raw.column_names)

    config = RewardConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        max_length=1024,
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Reward model saved to {out_dir}")


if __name__ == "__main__":
    main()
