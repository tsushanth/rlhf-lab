"""Classic RLHF: PPO-optimize the SFT policy against the reward model.

This is the expensive, historically unstable path — four models live in
memory at once (policy, reference policy for KL penalty, reward model,
value head), and training can reward-hack or KL-blow-up if the reward
model is weak or the KL coefficient is mistuned. That instability is the
actual reason the field moved toward DPO; this script exists to let you
feel it, not just read about it.
"""
import os
import torch
from transformers import AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOTrainer,
    PPOConfig,
)
from common import base_arg_parser, load_hh_rlhf, split_prompt_response, RESULTS_DIR


def main():
    parser = base_arg_parser("PPO fine-tune against the trained reward model")
    parser.add_argument("--sft-checkpoint", default=os.path.join(RESULTS_DIR, "sft"))
    parser.add_argument("--reward-model", default=os.path.join(RESULTS_DIR, "reward_model"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--kl-coef", type=float, default=0.05,
                         help="higher = stays closer to SFT policy, lower = more reward-seeking (and more prone to hacking)")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "ppo")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    ref_policy.config.pad_token_id = tokenizer.pad_token_id

    from transformers import AutoModelForSequenceClassification
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, num_labels=1, torch_dtype=torch.bfloat16
    )
    reward_model.eval()

    ppo_config = PPOConfig(
        learning_rate=1.4e-5,
        batch_size=16,
        mini_batch_size=4,
        init_kl_coef=args.kl_coef,
        target=6.0,
        steps=args.steps,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_policy,
        tokenizer=tokenizer,
    )

    raw = load_hh_rlhf(max_samples=args.max_samples)
    prompts = [split_prompt_response(ex["chosen"])[0] for ex in raw]

    device = trainer.accelerator.device
    reward_model.to(device)

    gen_kwargs = {"max_new_tokens": 128, "do_sample": True, "top_p": 0.9,
                  "pad_token_id": tokenizer.pad_token_id}

    batch_size = ppo_config.batch_size
    for step in range(args.steps):
        batch_prompts = prompts[(step * batch_size) % len(prompts):][:batch_size]
        if len(batch_prompts) < batch_size:
            batch_prompts += prompts[: batch_size - len(batch_prompts)]

        query_tensors = [
            tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            .input_ids[0].to(device)
            for p in batch_prompts
        ]
        response_tensors = trainer.generate(query_tensors, **gen_kwargs)
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        reward_inputs = tokenizer(
            [p + r for p, r in zip(batch_prompts, responses)],
            return_tensors="pt", truncation=True, max_length=1024, padding=True,
        ).to(device)
        with torch.no_grad():
            rewards = reward_model(**reward_inputs).logits.squeeze(-1)
        reward_list = [r for r in rewards]

        stats = trainer.step(query_tensors, response_tensors, reward_list)
        if step % 10 == 0:
            mean_reward = float(rewards.mean())
            kl = stats.get("objective/kl", float("nan"))
            print(f"step {step}/{args.steps}  mean_reward={mean_reward:.3f}  kl={kl:.3f}")

    trainer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"PPO checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
