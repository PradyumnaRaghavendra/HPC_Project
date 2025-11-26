"""
A*-PO Training Script for WebShop
Clean implementation using weighted SFT approach
"""
import argparse
import os
import sys
import json
import random
from pathlib import Path

from policy import PolicyModel, ReferenceModel
from webshop_env import WebShopEnvironment
from apo_trainer import APOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train A*-PO on WebShop')

    # Model
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--device', type=str, default='cuda')

    # Environment
    parser.add_argument('--num-products', type=int, default=100,
                       help='Number of products in WebShop')

    # Training
    parser.add_argument('--num-steps', type=int, default=200,
                       help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Number of instructions per batch')

    # A*-PO hyperparameters
    parser.add_argument('--beta', type=float, default=0.5,
                       help='A*-PO beta parameter for V* smoothing')
    parser.add_argument('--v-star-samples', type=int, default=8,
                       help='Number of samples for V* computation')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--kl-coef', type=float, default=0.02,
                       help='KL regularization coefficient')
    parser.add_argument('--adv-clip', type=float, default=3.0,
                       help='Advantage clipping value')

    # Sampling
    parser.add_argument('--temperature', type=float, default=0.9,
                       help='Sampling temperature')
    parser.add_argument('--max-episode-steps', type=int, default=15,
                       help='Max steps per episode')

    # Evaluation
    parser.add_argument('--eval-frequency', type=int, default=20,
                       help='Evaluate every N steps')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    # Checkpointing
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--save-frequency', type=int, default=50,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    env: WebShopEnvironment,
    policy: PolicyModel,
    num_episodes: int,
    max_steps: int,
    temperature: float
):
    """Run evaluation episodes."""
    policy.eval()

    rewards = []
    success_count = 0

    print(f"\n{'='*60}")
    print(f"EVALUATION ({num_episodes} episodes)")
    print(f"{'='*60}")

    for ep in range(num_episodes):
        obs, info = env.reset()
        instruction = info['instruction']

        total_reward = 0.0
        done = False
        step = 0
        actions = []

        while not done and step < max_steps:
            action = policy.generate_action(
                instruction=instruction,
                observation=obs,
                previous_actions=actions,
                temperature=temperature * 0.8,  # Lower temp for eval
                do_sample=True
            )

            actions.append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        raw_reward = info.get('raw_reward', 0.0)
        success = raw_reward >= 0.9

        rewards.append(total_reward)
        if success:
            success_count += 1

        print(f"  Episode {ep+1}: Reward={total_reward:.3f}, Success={success}, Steps={step}")

    policy.train()

    return {
        'mean_reward': sum(rewards) / len(rewards),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes
    }


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("A*-PO Training for WebShop")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training steps: {args.num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"="*60 + "\n")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Initialize environment
    print("Initializing WebShop environment...")
    env = WebShopEnvironment(num_products=args.num_products)

    # Initialize models
    print("Loading policy model...")
    policy = PolicyModel(model_name=args.model, device=args.device)

    print("Loading reference model...")
    ref_model = ReferenceModel(model_name=args.model, device=args.device)

    # Create config
    config = {
        'apo': {
            'beta': args.beta,
            'v_star_samples': args.v_star_samples,
            'learning_rate': args.learning_rate,
            'kl_coef': args.kl_coef,
            'adv_clip': args.adv_clip,
            'clip_grad_norm': 1.0,
            'adaptive_vstar': True
        },
        'model': {
            'max_steps': args.max_episode_steps,
            'max_length': 512
        },
        'sampling': {
            'temperature': args.temperature,
            'top_p': 0.95
        }
    }

    # Initialize trainer
    print("Initializing A*-PO trainer...")
    trainer = APOTrainer(
        policy_model=policy,
        ref_model=ref_model,
        env=env,
        config=config
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        policy.load(args.resume)
        vstar_cache_path = Path(args.resume).parent / 'vstar_cache.json'
        if vstar_cache_path.exists():
            trainer.load_vstar_cache(str(vstar_cache_path))

    # Load available instructions from WebShop data (like TinyZero loads dataset)
    print("Loading available instructions from WebShop...")

    # Load directly from WebShop's goal file (efficient!)
    webshop_path = os.environ.get('WEBSHOP_PATH', '/root/WebShop')
    goals_file = os.path.join(webshop_path, 'data', 'items_human_ins.json')

    try:
        with open(goals_file, 'r') as f:
            goals_data = json.load(f)

        print(f"  Goals data type: {type(goals_data)}")
        if isinstance(goals_data, list):
            print(f"  It's a list with {len(goals_data)} items")
            if goals_data:
                print(f"  First item type: {type(goals_data[0])}")
                print(f"  First item keys: {list(goals_data[0].keys()) if isinstance(goals_data[0], dict) else 'not a dict'}")
        elif isinstance(goals_data, dict):
            print(f"  It's a dict with {len(goals_data)} keys")
            first_key = list(goals_data.keys())[0] if goals_data else None
            if first_key:
                print(f"  First value type: {type(goals_data[first_key])}")
                if isinstance(goals_data[first_key], list) and goals_data[first_key]:
                    print(f"  First list item type: {type(goals_data[first_key][0])}")
                    if isinstance(goals_data[first_key][0], dict):
                        print(f"  First list item keys: {list(goals_data[first_key][0].keys())}")

        # Check if it's a list or dict
        if isinstance(goals_data, list):
            # It's a list of items
            available_instructions = list(set([item['instruction'] for item in goals_data if 'instruction' in item]))
        else:
            # It's a dict with IDs as keys, and each value is a list of items
            # Collect ALL instructions (not just first from each product)
            instructions = []
            for key, value_list in goals_data.items():
                if isinstance(value_list, list):
                    for item in value_list:
                        if isinstance(item, dict) and 'instruction' in item:
                            instructions.append(item['instruction'])
            available_instructions = list(set(instructions))

        print(f"  ✓ Loaded {len(available_instructions)} unique instructions from {goals_file}")
    except Exception as e:
        # Fallback: sample from environment (slower but works)
        print(f"  Could not load goals file ({e}), sampling from environment...")
        available_instructions = []
        seen = set()
        for _ in range(200):  # Increase attempts to get more instructions
            obs, info = env.reset()
            instruction = info['instruction']
            if instruction not in seen:
                available_instructions.append(instruction)
                seen.add(instruction)
            if len(available_instructions) >= 50:  # Stop after 50 unique instructions
                break

    print(f"✓ Loaded {len(available_instructions)} instructions")

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    best_success_rate = 0.0

    for step in range(args.num_steps):
        print(f"\n--- Step {step+1}/{args.num_steps} ---")

        # Sample batch from available instructions (like TinyZero's dataloader)
        batch_instructions = random.sample(available_instructions, min(args.batch_size, len(available_instructions)))

        # Training step
        loss, stats = trainer.train_step(batch_instructions)

        # Log
        print(f"  Loss: {loss:.4f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f}")
        print(f"  Avg V*: {stats['avg_v_star']:.3f}")
        print(f"  Avg Advantage: {stats['avg_advantage']:.3f}")
        print(f"  Weight mean/std: {stats['weight_mean']:.2f} / {stats['weight_std']:.2f}")
        print(f"  Trajectories generated: {stats['num_trajectories']}")

        # Evaluation
        if (step + 1) % args.eval_frequency == 0:
            eval_stats = evaluate(
                env=env,
                policy=policy,
                num_episodes=args.eval_episodes,
                max_steps=args.max_episode_steps,
                temperature=args.temperature
            )

            print(f"\n  Evaluation Results:")
            print(f"    Mean Reward: {eval_stats['mean_reward']:.3f}")
            print(f"    Success Rate: {eval_stats['success_rate']:.1%}")

            # Save best model
            if eval_stats['success_rate'] > best_success_rate:
                best_success_rate = eval_stats['success_rate']
                best_path = output_dir / 'best_model'
                policy.save(str(best_path))
                trainer.save_vstar_cache(str(best_path / 'vstar_cache.json'))
                print(f"    ✓ New best model saved! (success rate: {best_success_rate:.1%})")

        # Save checkpoint
        if (step + 1) % args.save_frequency == 0:
            checkpoint_path = output_dir / f'checkpoint_{step+1}'
            policy.save(str(checkpoint_path))
            trainer.save_vstar_cache(str(checkpoint_path / 'vstar_cache.json'))
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    final_eval = evaluate(
        env=env,
        policy=policy,
        num_episodes=args.eval_episodes * 2,  # More episodes
        max_steps=args.max_episode_steps,
        temperature=args.temperature
    )

    print(f"\nFinal Results:")
    print(f"  Mean Reward: {final_eval['mean_reward']:.3f}")
    print(f"  Success Rate: {final_eval['success_rate']:.1%}")
    print(f"  Best Success Rate: {best_success_rate:.1%}")

    # Save final model
    final_path = output_dir / 'final_model'
    policy.save(str(final_path))
    trainer.save_vstar_cache(str(final_path / 'vstar_cache.json'))

    print(f"\n✓ Training complete! Models saved to {output_dir}")


if __name__ == '__main__':
    main()
