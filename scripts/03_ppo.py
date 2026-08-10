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
    parser.add_argument("--kl-coef", type=float, default=0.2,
                         help="higher = stays closer to SFT policy, lower = more reward-seeking (and more prone to hacking)")
    parser.add_argument("--resume-from", default=None,
                         help="path to a periodic checkpoint (e.g. results/ppo_checkpoint) to resume the "
                              "policy from after an interrupted run, instead of starting fresh from SFT")
    parser.add_argument("--start-step", type=int, default=0,
                         help="step count to resume the batch cursor / step counter from, matching --resume-from")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "ppo")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_source = args.resume_from if args.resume_from else args.sft_checkpoint
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_source, torch_dtype=torch.bfloat16)
    policy.config.pad_token_id = tokenizer.pad_token_id
    # The reference policy for the KL penalty always stays anchored to the
    # original SFT checkpoint, even when resuming -- it's the fixed point
    # PPO measures drift against, not something that should itself resume.
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
        # mini_batch_size=4 with trl's default ppo_epochs=4 meant 4 gradient
        # steps per 4-sample slice -> the policy overfit each mini-batch fast
        # enough to blow the importance-sampling ratio past trl's own
        # sanity threshold ("average ratio of batch exceeds threshold 10"),
        # which is what drove objective/kl negative. Raising mini_batch_size
        # to reduce that noise OOMs this GPU with 4 bf16 models resident
        # (policy+value head, ref policy, reward model), so instead cut
        # ppo_epochs to 2 -- fewer gradient steps per mini-batch means less
        # room to overfit it even at the same mini_batch_size.
        mini_batch_size=4,
        ppo_epochs=2,
        init_kl_coef=args.kl_coef,
        target=6.0,
        steps=args.steps,
        # Diagnosed via ppo/val/var_explained = -0.92 on a fresh run: the
        # value head is randomly initialized (AutoModelForCausalLMWithValueHead
        # adds it fresh on top of the SFT checkpoint) and default vf_coef=0.1
        # barely trains it relative to the policy loss, so early advantage
        # estimates are computed against a near-random baseline -- garbage
        # advantages, garbage policy gradient, which is what showed up as
        # mean_reward trending down and KL oscillating wildly rather than
        # converging. Raising vf_coef gives the value function's loss much
        # more weight in the combined objective so it actually learns fast
        # enough to become a useful baseline within the run's step budget.
        vf_coef=1.0,
        # trl's PPOConfig defaults max_grad_norm to None (no clipping). A
        # 500-step run with vf_coef=1.0 hit a single outlier batch at step
        # 211 (policy_loss spiked to 3.1) whose unclipped gradient pushed
        # the policy weights to NaN -- generation then produced NaN/inf
        # softmax probabilities and the process crashed outright. Standard
        # PPO implementations (e.g. CleanRL, OpenAI's baselines) clip to 0.5
        # by default specifically to survive this kind of single-batch
        # outlier; trl doesn't turn it on for you.
        max_grad_norm=0.5,
        # Normalize+clip the reward model's raw score before it hits the
        # advantage computation. The RM's outputs aren't calibrated to any
        # particular scale (observed mean ~-3, std ~1.5, pairwise accuracy
        # only 68%) -- without this, an occasional high-magnitude/noisy
        # score can dominate the advantage and cause a large, destabilizing
        # policy update.
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=3.0,
        # Skip the PPO update entirely on batches where KL already exceeds
        # 1.5x target, instead of pushing the policy further off distribution.
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
    # In-memory snapshot of the last known-good policy weights, refreshed
    # every 10 steps. A rare batch can produce an astronomically large
    # importance-sampling ratio (observed: ~2e18, then literally inf) purely
    # from bf16 precision limits in the log-prob/ratio math itself -- this
    # happens in the forward pass, before gradients even exist, so gradient
    # clipping and reward sanitization can't catch it. trl's own "skipping
    # batch" guard fires but doesn't fully prevent the corruption; the next
    # generate() call then crashes on NaN weights. Roll back in memory
    # instead of crashing the whole run over one bad batch.
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
        # Two full 500-step runs crashed to NaN despite gradient clipping --
        # once at step 211 (mid-run) and once at step 499 (the very last
        # step, both non-deterministic w.r.t. step count). Clipping bounds
        # gradient *norm*, but a single inf/nan reward from the reward model
        # on some degenerate/edge-case generation propagates through the
        # loss untouched by norm clipping (nan/inf survive clipping). Sanitize
        # at the source instead of hoping it never happens again.
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
        reward_list = [r for r in rewards]

        stats = trainer.step(query_tensors, response_tensors, reward_list)
        mean_reward = float(rewards.mean())
        kl = float(stats.get("ppo/policy/policykl", float("nan")))
        var_explained = float(stats.get("ppo/val/var_explained", float("nan")))
        policy_loss = float(stats.get("ppo/loss/policy", float("nan")))

        if any(map(lambda x: x != x, (kl, var_explained, policy_loss))):  # nan-safe check, no numpy import needed
            print(f"step {step}/{args.steps}  NaN detected (kl={kl} policy_loss={policy_loss}) "
                  f"-- rolling back to last known-good weights and continuing", flush=True)
            policy.load_state_dict({k: v.to(device) for k, v in last_good_state.items()})
        else:
            print(f"step {step}/{args.steps}  mean_reward={mean_reward:.3f}  kl={kl:.4f}  "
                  f"val_var_explained={var_explained:.3f}  policy_loss={policy_loss:.4f}", flush=True)
            if step % 10 == 0:
                last_good_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        # Both prior full runs completed all steps and only went NaN at the
        # very end (or crashed mid-run before ever reaching save_pretrained)
        # -- with saving only happening after the loop, either failure mode
        # threw away hours of otherwise-good training. Checkpoint into a
        # rotating slot periodically so the last known-good state survives
        # a late crash; overwrite in place rather than keeping every
        # snapshot; only 1.5B params so overhead is well within budget.
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
