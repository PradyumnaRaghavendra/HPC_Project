"""
A*-PO Training Script for WebShop
Two-stage training: Stage 1 (Offline) -> Stage 2 (Online)
"""
import sys
sys.path.append('src')

import torch
import argparse
from pathlib import Path

from environments.webshop_env import WebShopEnvironment
from models.policy import RAGENPolicy
from utils.value_cache import OptimalValueCache
from trainers.stage1_offline import Stage1OfflineTrainer
from trainers.stage2_online import Stage2OnlineTrainer

def main(args):
    """Main A*-PO training function."""
    
    print("\n" + "="*60)
    print("A*-PO Training on WebShop")
    print("="*60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create directories
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize environment
    print(f"\nInitializing WebShop environment...")
    env = WebShopEnvironment(num_products=args.num_products)
    
    # Initialize policy
    print(f"\nInitializing policy: {args.model_name}")
    policy = RAGENPolicy(
        model_name=args.model_name,
        max_length=args.max_length,
        device=device
    )
    
    # Initialize value cache
    print(f"\nInitializing V* cache...")
    value_cache = OptimalValueCache(
        cache_file=f"{args.checkpoint_dir}/v_star_cache.json"
    )
    
    # STAGE 1: Offline Value Estimation
    if not args.skip_stage1:
        print(f"\n{'='*60}")
        print("STAGE 1: Offline Optimal Value Estimation")
        print(f"{'='*60}")
        
        stage1_trainer = Stage1OfflineTrainer(
            env=env,
            policy=policy,
            value_cache=value_cache,
            k_samples=args.k_samples,
            max_steps=args.max_steps,
            temperature=args.temperature
        )
        
        stage1_trainer.collect_optimal_values(
            num_instructions=args.num_instructions_stage1,
            verbose=True
        )
    else:
        print("\nSkipping Stage 1 (using existing V* cache)")
    
    # STAGE 2: Online Policy Optimization
    print(f"\n{'='*60}")
    print("STAGE 2: Online Policy Optimization")
    print(f"{'='*60}")
    
    stage2_trainer = Stage2OnlineTrainer(
        env=env,
        policy=policy,
        value_cache=value_cache,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    stage2_trainer.train(
        num_iterations=args.num_iterations,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        temperature=args.temperature
    )
    
    print("\n✓ A*-PO training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A*-PO on WebShop")
    
    # Environment
    parser.add_argument("--num_products", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=20)
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_length", type=int, default=512)
    
    # Stage 1: Offline
    parser.add_argument("--skip_stage1", action="store_true",
                       help="Skip Stage 1 if V* cache exists")
    parser.add_argument("--num_instructions_stage1", type=int, default=50,
                       help="Number of instructions for Stage 1")
    parser.add_argument("--k_samples", type=int, default=64,
                       help="Trajectories per instruction in Stage 1")
    
    # Stage 2: Online
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate (reduced from 1e-5 for stability)")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="A*-PO advantage scaling coefficient (increased for stronger signal)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (reduced for more focused exploration)")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument("--save_frequency", type=int, default=50)
    
    args = parser.parse_args()
    main(args)
