"""Classic RLHF: PPO-optimize the SFT policy against the reward model.

This is the expensive, historically unstable path — four models live in
memory at once (policy, reference policy for KL penalty, reward model,
value head), and training can reward-hack or KL-blow-up if the reward
model is weak or the KL coefficient is mistuned.

**Enhanced over 03_ppo.py:**
- `--save-generations-every N`: periodically generate responses to a fixed
  set of snapshot prompts and append to a JSONL file. This lets you
  inspect *what* the policy is producing as it degrades, not just the KL
  numbers.
- `--snapshot-prompts`: optional JSONL of custom prompts; if omitted, uses
  the first 5 HH-RLHF prompts.

All stability fixes from 03_ppo.py are preserved (vf_coef=1.0, grad clipping,
score scaling, in-memory rollback, periodic checkpoints).
"""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from common import base_arg_parser, load_hh_rlhf, split_prompt_response, RESULTS_DIR


def generate_samples(model, tokenizer, prompts, device, max_new_tokens=128):
    """Generate one response per prompt. Returns list of strings."""
    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append(text)
    return results


def main():
    parser = base_arg_parser("PPO fine-tune (enhanced with generation snapshots)")
    parser.add_argument("--sft-checkpoint", default=os.path.join(RESULTS_DIR, "sft"))
    parser.add_argument("--reward-model", default=os.path.join(RESULTS_DIR, "reward_model"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--kl-coef", type=float, default=0.2)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--save-generations-every", type=int, default=50,
                        help="Every N steps, generate responses to snapshot prompts and save to JSONL")
    parser.add_argument("--snapshot-prompts", default=None,
                        help="JSONL file with {'prompt': ...} lines. If omitted, uses first 5 HH-RLHF prompts.")
    parser.add_argument("--generations-out", default=os.path.join(RESULTS_DIR, "ppo_generations.jsonl"),
                        help="JSONL file to append snapshot generations to")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "ppo")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Snapshot prompts
    # -----------------------------------------------------------------------
    if args.snapshot_prompts and os.path.exists(args.snapshot_prompts):
        with open(args.snapshot_prompts) as f:
            snapshot_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    else:
        raw_snap = load_hh_rlhf(max_samples=5)
        snapshot_prompts = [split_prompt_response(ex["chosen"])[0] for ex in raw_snap]
    print(f"[snapshots] Using {len(snapshot_prompts)} snapshot prompts")
    # Ensure generations file dir exists
    os.makedirs(os.path.dirname(args.generations_out) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_source = args.resume_from if args.resume_from else args.sft_checkpoint
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_source, torch_dtype=torch.bfloat16)
    policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_checkpoint, torch_dtype=torch.bfloat16)
    ref_policy.config.pad_token_id = tokenizer.pad_token_id

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, num_labels=1, torch_dtype=torch.bfloat16
    )
    reward_model.eval()

    ppo_config = PPOConfig(
        learning_rate=1.4e-5,
        batch_size=16,
        mini_batch_size=4,
        ppo_epochs=2,
        init_kl_coef=args.kl_coef,
        target=6.0,
        steps=args.steps,
        vf_coef=1.0,
        max_grad_norm=0.5,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=3.0,
        early_stopping=True,
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
    last_good_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    for step in range(args.start_step, args.steps):
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
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
        reward_list = [r for r in rewards]

        stats = trainer.step(query_tensors, response_tensors, reward_list)
        mean_reward = float(rewards.mean())
        kl = float(stats.get("ppo/policy/policykl", float("nan")))
        var_explained = float(stats.get("ppo/val/var_explained", float("nan")))
        policy_loss = float(stats.get("ppo/loss/policy", float("nan")))

        if any(map(lambda x: x != x, (kl, var_explained, policy_loss))):
            print(f"step {step}/{args.steps}  NaN detected (kl={kl} policy_loss={policy_loss}) "
                  f"-- rolling back to last known-good weights and continuing", flush=True)
            policy.load_state_dict({k: v.to(device) for k, v in last_good_state.items()})
        else:
            print(f"step {step}/{args.steps}  mean_reward={mean_reward:.3f}  kl={kl:.4f}  "
                  f"val_var_explained={var_explained:.3f}  policy_loss={policy_loss:.4f}", flush=True)
            if step % 10 == 0:
                last_good_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        # -------------------------------------------------------------------
        # Periodic generation snapshots
        # -------------------------------------------------------------------
        if step > 0 and step % args.save_generations_every == 0:
            snap_texts = generate_samples(policy, tokenizer, snapshot_prompts, device)
            snap_record = {
                "step": step,
                "kl": kl,
                "mean_reward": mean_reward,
                "policy_loss": policy_loss,
                "var_explained": var_explained,
                "generations": [
                    {"prompt": p, "response": r}
                    for p, r in zip(snapshot_prompts, snap_texts)
                ],
            }
            with open(args.generations_out, "a") as gf:
                gf.write(json.dumps(snap_record) + "\n")
            print(f"  [snapshot saved -> {args.generations_out}]", flush=True)

        # Periodic checkpoint
        if step > 0 and step % 100 == 0:
            ckpt_dir = out_dir + "_checkpoint"
            trainer.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  [checkpoint saved at step {step} -> {ckpt_dir}]", flush=True)

    trainer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"PPO checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
