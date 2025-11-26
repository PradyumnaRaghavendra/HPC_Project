"""
Main training script for RAGEN on WebShop
Run with: python train.py
"""
import sys
sys.path.append('src')

import torch
import argparse
from pathlib import Path

from environments.webshop_env import WebShopEnvironment
from models.policy import RAGENPolicy
from models.value import ValueModel
from utils.rollouts import RolloutCollector
from trainers.ragen_trainer import RAGENTrainer

def main(args):
    """Main training function."""
    
    print("\n" + "="*60)
    print("RAGEN Training on WebShop")
    print("="*60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create directories
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize environment
    print(f"\nInitializing WebShop environment...")
    print(f"  Num products: {args.num_products}")
    env = WebShopEnvironment(num_products=args.num_products)
    
    # Initialize models
    print(f"\nInitializing models...")
    print(f"  Model: {args.model_name}")
    print(f"  Max length: {args.max_length}")
    
    policy = RAGENPolicy(
        model_name=args.model_name,
        max_length=args.max_length,
        device=device
    )
    
    value_model = ValueModel(
        model_name=args.model_name,
        max_length=args.max_length,
        device=device
    )
    
    # Initialize rollout collector
    print(f"\nInitializing rollout collector...")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Gamma: {args.gamma}")
    
    rollout_collector = RolloutCollector(
        env=env,
        policy=policy,
        value_model=value_model,
        max_steps=args.max_steps,
        gamma=args.gamma
    )
    
    # Initialize trainer
    print(f"\nInitializing trainer...")
    print(f"  Policy LR: {args.learning_rate}")
    print(f"  Value LR: {args.value_lr}")
    print(f"  Clip epsilon: {args.clip_epsilon}")
    print(f"  KL coeff: {args.kl_coeff}")
    print(f"  Entropy coeff: {args.entropy_coeff}")
    
    trainer = RAGENTrainer(
        policy=policy,
        value_model=value_model,
        rollout_collector=rollout_collector,
        learning_rate=args.learning_rate,
        value_lr=args.value_lr,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    print(f"\nStarting training...")
    trainer.train(
        num_iterations=args.num_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        ppo_epochs=args.ppo_epochs,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        temperature=args.temperature
    )
    
    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAGEN on WebShop")
    
    # Environment
    parser.add_argument("--num_products", type=int, default=100,
                       help="Number of products in WebShop (100, 1000, or 'all')")
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Maximum steps per episode")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training
    parser.add_argument("--num_iterations", type=int, default=500,
                       help="Number of training iterations")
    parser.add_argument("--episodes_per_iteration", type=int, default=4,
                       help="Episodes to collect per iteration")
    parser.add_argument("--ppo_epochs", type=int, default=3,
                       help="PPO update epochs per iteration")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Policy learning rate")
    parser.add_argument("--value_lr", type=float, default=3e-5,
                       help="Value model learning rate")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                       help="PPO clipping parameter")
    parser.add_argument("--kl_coeff", type=float, default=0.1,
                       help="KL penalty coefficient")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                       help="Entropy bonus coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    
    # Logging & Checkpointing
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Model checkpoint directory")
    parser.add_argument("--eval_frequency", type=int, default=10,
                       help="Evaluate every N iterations")
    parser.add_argument("--save_frequency", type=int, default=50,
                       help="Save checkpoint every N iterations")
    
    args = parser.parse_args()
    main(args)