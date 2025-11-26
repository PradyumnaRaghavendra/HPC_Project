"""
Training script for RAGEN (Multi-turn agent training)
Based on TinyZero's train.py but adapted for multi-turn environments
"""
import argparse
import yaml
import torch
import sys
from pathlib import Path
from tqdm import tqdm
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tinyzero.models import PolicyModel, ReferenceModel
from tinyzero.utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter
from tinyzero.ppo_trainer import PPOTrainer
from ragen.agent_trainer import MultiTurnAPOTrainer
from ragen.multi_turn_ppo_trainer import MultiTurnPPOTrainer
from ragen.webshop_data_real import create_webshop_dataloaders
from ragen.environments import WebShopEnvironment
from ragen.curriculum import CurriculumManager

#  Action sanitizer import (required by WebShop env to normalize actions)
try:
    from ragen.action_sanitizer import sanitize
except Exception:
    # Fallback (no-op) if sanitizer module isn't present; training will still run
    def sanitize(text, fallback_query=None):
        return (text or "").strip()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RAGEN with A*PO')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ragen_webshop.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/ragen',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (fewer steps, faster)'
    )
    parser.add_argument(
        '--skip-initial-eval',
        action='store_true',
        help='Skip initial evaluation (faster startup)'
    )
    return parser.parse_args()


class RAGENTrainer:
    """RAGEN trainer - extends TinyZero training for multi-turn agents"""
    
    def __init__(self, config: dict, output_dir: str):
        """
        Initialize RAGEN trainer
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.output_dir / 'training.log'
        self.metrics_file = self.output_dir / 'metrics.json'
        
        # Initialize metrics storage
        self.metrics_history = {
            'train_loss': [],
            'train_reward': [],
            'train_success_rate': [],
            'train_avg_turns': [],
            'eval_success_rate': [],
            'steps': []
        }
        
        # Set seed
        set_seed(config.get('seed', 42))
        
        # Create dataloaders
        env_type = config.get('environment', {}).get('type', 'webshop')
        if env_type == 'maze':
            print("Maze environment - creating task dataloaders")
            # Create maze task dataloaders
            from torch.utils.data import DataLoader, Dataset
            class MazeDataset(Dataset):
                def __init__(self, size):
                    self.size = size
                def __len__(self):
                    return self.size
                def __getitem__(self, idx):
                    # Return a proper maze task with instruction
                    return {
                        'instruction': "Task: Get from (0, 0) to (4, 4)",
                        'asin': f'maze_{idx}',  # Dummy ID for compatibility
                    }
            # Custom collate to keep dicts as list (don't batch them)
            def dict_collate(batch):
                return batch  # Return list of dicts as-is

            self.train_loader = DataLoader(
                MazeDataset(config.get('data', {}).get('train_size', 10)),
                batch_size=1,
                collate_fn=dict_collate
            )
            self.eval_loader = DataLoader(
                MazeDataset(config.get('data', {}).get('eval_size', 5)),
                batch_size=1,
                collate_fn=dict_collate
            )
        elif env_type == 'medium':
            print("Creating Medium WebShop dataloaders...")
            from ragen.medium_webshop_dataloader import create_medium_webshop_dataloaders
            self.train_loader, self.eval_loader = create_medium_webshop_dataloaders(config)
        else:
            print("Creating WebShop dataloaders...")
            self.train_loader, self.eval_loader = create_webshop_dataloaders(config)

        if env_type != 'maze':
            print(f"Train samples: {len(self.train_loader.dataset)}")
            print(f"Eval samples: {len(self.eval_loader.dataset)}")
        
        # Initialize models
        print("Loading models...")
        self.policy = PolicyModel(
            config['model']['name'],
            device=config['model']['device']
        )
        self.ref_model = ReferenceModel(
            config['model']['ref_model'],
            device=config['model']['device']
        )
        print("Models loaded successfully!")

        # Initialize curriculum manager BEFORE environment
        self.curriculum = CurriculumManager(config)

        # Initialize environment (with curriculum's product count)
        print("Initializing environment...")
        env_type = config.get('environment', {}).get('type', 'webshop')

        if env_type == 'simple':
            from ragen.environments.simple_webshop import SimpleWebShopEnvironment
            self.environment = SimpleWebShopEnvironment(config)
            print("✓ SimpleShop environment ready!")
        elif env_type == 'medium':
            from ragen.environments.medium_webshop import MediumWebShopEnvironment
            self.environment = MediumWebShopEnvironment(config)
            print("✓ MediumWebShop environment ready!")
        elif env_type == 'maze':
            from ragen.environments.simple_maze import SimpleMazeEnvironment
            self.environment = SimpleMazeEnvironment(config)
            print("✓ SimpleMaze environment ready!")
        else:
            self.environment = WebShopEnvironment(config)
            print("✓ WebShop environment ready!")

        # Store for environment reinitialization
        self.env_type = env_type

        # Initialize multi-turn trainer (PPO or A*-PO)
        trainer_type = config.get('trainer', {}).get('type', 'apo')  # Default to A*-PO

        if trainer_type == 'ppo':
            print("\n🎯 Using Multi-Turn PPO Trainer (with trained critic)")
            print("   Following RAGEN paper's specifications")
            print("   - Trained critic for advantage estimation")
            print("   - GAE with γ=1.0, λ=1.0")
            print("   - Multi-turn environment interaction")
            self.trainer = MultiTurnPPOTrainer(
                self.policy,
                self.ref_model,
                config,
                self.environment
            )
        else:
            print("\n🎯 Using A*-PO Trainer (critic-free)")
            self.trainer = MultiTurnAPOTrainer(
                self.policy,
                self.ref_model,
                config,
                self.environment
            )

        # Keep backward compatibility
        self.apo_trainer = self.trainer
        
        # Training state
        self.global_step = 0
        self.best_success_rate = 0.0
        self.skip_initial_eval = False  # Will be set by main()
        
        # Meters for tracking
        self.loss_meter = AverageMeter()
        self.reward_meter = AverageMeter()
    
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


    def pretrain_on_demos(self, num_steps=30):
        """
        Behavioral cloning on expert demonstrations.
        Teaches the model successful action sequences.
        """
        print("\n" + "="*60)
        print("🎓 PHASE 1: LEARNING FROM EXPERT DEMONSTRATIONS")
        print("="*60)
        
        # Load expert demos based on environment type
        if self.env_type == 'medium':
            from ragen.medium_expert_demos import get_medium_expert_demos
            all_demos = get_medium_expert_demos()
            print(f"  Training on {len(all_demos)} medium WebShop expert demos...")
        elif self.env_type == 'maze':
            from ragen.maze_expert_demos import get_maze_expert_demos
            all_demos = get_maze_expert_demos()
            print(f"  Training on {len(all_demos)} maze expert demos...")
        else:
            from ragen.expert_demos import get_expert_demos, get_partial_demos
            full_demos = get_expert_demos()
            partial_demos = get_partial_demos()
            all_demos = full_demos + partial_demos
            print(f"  Training on {len(all_demos)} expert trajectories...")
            print(f"  {len(full_demos)} perfect demos, {len(partial_demos)} partial demos")
        
        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=0.0001  # Higher LR for faster imitation learning
        )
        
        demo_losses = []
        
        for step in range(num_steps):
            epoch_loss = 0.0
            
            for demo in all_demos:
                # Build prompt (simple, natural language)
                prompt = f"Task: {demo['instruction']}\n\nWhat should you do?"
                
                # Target is the first action (most critical!)
                # We focus on teaching the FIRST action: search[...]
                target_action = demo['actions'][0]
                
                # Tokenize prompt
                prompt_enc = self.policy.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Tokenize target action
                target_enc = self.policy.tokenizer(
                    target_action,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                # Move to device
                device = self.policy.model.device
                prompt_ids = prompt_enc['input_ids'].to(device)
                prompt_mask = prompt_enc['attention_mask'].to(device)
                target_ids = target_enc['input_ids'].to(device)
                
                # Concatenate prompt + target
                input_ids = torch.cat([prompt_ids, target_ids], dim=1)
                attention_mask = torch.cat([
                    prompt_mask,
                    torch.ones_like(target_ids)
                ], dim=1)
                
                # Create labels (mask prompt, only learn on action)
                labels = input_ids.clone()
                labels[:, :prompt_ids.shape[1]] = -100  # Mask prompt
                
                # Forward pass
                outputs = self.policy.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                optimizer.step()
            
            avg_loss = epoch_loss / len(all_demos)
            demo_losses.append(avg_loss)
            
            if step % 5 == 0:
                print(f"  Step {step}/{num_steps}: Loss={avg_loss:.4f}")
        
        print(f"\n  ✓ Pre-training complete!")
        print(f"  Initial loss: {demo_losses[0]:.4f}")
        print(f"  Final loss: {demo_losses[-1]:.4f}")
        print(f"  Improvement: {demo_losses[0] - demo_losses[-1]:.4f}")
        print("="*60 + "\n")
        
        # Test generation
        print(" Testing learned behavior:")
        test_prompts = [
            "Task: find blue headphones\n\nWhat should you do?",
            "Task: find women sweaters\n\nWhat should you do?",
        ]
        
        self.policy.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                output = self.policy.generate(
                    [prompt],
                    max_length=50,
                    temperature=0.1,
                    do_sample=True
                )[0]
                print(f"  {prompt.split(':')[1].split('What')[0].strip()}")
                print(f"    → {output[:60]}")
        self.policy.train()
        
        print("\n" + "="*60)
        print("🚀 PHASE 2: REINFORCEMENT LEARNING")
        print("="*60 + "\n")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.policy.train()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}"
        )
        
        for batch_idx, batch in pbar:
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                break

            # BC REGULARIZATION **RE-ENABLED WITH REAL DEMOS**
            # Mix 20% real expert demos into training batch to prevent forgetting
            batch = self.apo_trainer.inject_expert_demos(batch, ratio=0.2)

            # Training step (MULTI-TURN!)
            try:
                loss, metrics = self.apo_trainer.train_step_multiturn(batch)
                
                # Update meters
                self.loss_meter.update(loss)
                self.reward_meter.update(metrics['avg_reward'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{self.loss_meter.avg:.4f}",
                    'reward': f"{self.reward_meter.avg:.3f}",
                    'success': f"{metrics.get('success_rate', 0.0):.2%}",
                    'step': self.global_step
                })
                
            except Exception as e:
                self.log(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Logging
            if self.global_step % self.config['logging']['log_every'] == 0:
                success_rate = metrics.get('success_rate', 0.0)
                self.log(
                    f"Step {self.global_step}: "
                    f"Loss={self.loss_meter.avg:.4f}, "
                    f"Reward={self.reward_meter.avg:.3f}, "
                    f"Success={success_rate:.2%}"
                )

                # Store metrics
                self.metrics_history['train_loss'].append(self.loss_meter.avg)
                self.metrics_history['train_reward'].append(self.reward_meter.avg)
                self.metrics_history['train_success_rate'].append(success_rate)
                self.metrics_history['train_avg_turns'].append(metrics.get('avg_turns', 0))
                self.metrics_history['steps'].append(self.global_step)
                
                # Reset meters
                self.loss_meter.reset()
                self.reward_meter.reset()
            
            # Evaluation (only if eval_every is reasonable)
            if (self.global_step > 0 and
                self.global_step % self.config['training']['eval_every'] == 0 and 
                self.config['training']['eval_every'] < 100):
                self.log(f"\n{'='*50}")
                self.log(f"Running evaluation at step {self.global_step}...")
                
                eval_success = self.evaluate()

                # Store eval metrics
                self.metrics_history['eval_success_rate'].append(eval_success)

                # Check curriculum progression (Option C)
                self.check_curriculum_progress(eval_success)

                # Log results
                self.log(f"Eval - Success Rate: {eval_success:.2%}")
                self.log(f"   {self.curriculum.get_progress_str()}")
                
                # Save best model
                if eval_success > self.best_success_rate:
                    self.best_success_rate = eval_success
                    self.log(f"New best success rate: {self.best_success_rate:.2%}")
                    save_checkpoint(
                        self.apo_trainer.policy.model,
                        self.apo_trainer.optimizer,
                        self.global_step,
                        self.output_dir / 'best_model.pt',
                        success_rate=self.best_success_rate
                    )
                
                self.log(f"{'='*50}\n")
                
                # Back to training mode
                self.policy.train()
            
            # Checkpointing (only if save_every is reasonable)
            if (self.global_step > 0 and
                self.global_step % self.config['training']['save_every'] == 0 and
                self.config['training']['save_every'] < 100):
                checkpoint_path = self.output_dir / f'checkpoint_{self.global_step}.pt'
                save_checkpoint(
                    self.apo_trainer.policy.model,
                    self.apo_trainer.optimizer,
                    self.global_step,
                    checkpoint_path
                )
                self.log(f"Saved checkpoint to {checkpoint_path}")
            
            # Save metrics periodically
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.save_metrics()
            
            self.global_step += 1
    
    def evaluate(self) -> float:
        """
        Evaluate on eval set

        Returns:
            Success rate
        """
        self.policy.eval()

        successes = 0
        total = 0

        print(f"\n Evaluating on {len(self.eval_loader.dataset)} tasks...")
        print("="*80)
        print("DETAILED EVALUATION LOGGING - First 5 tasks")
        print("="*80)

        # Pull commonly used knobs from config
        model_cfg = self.config.get('model', {})
        sampling_cfg = self.config.get('sampling', {})
        use_min_new_tokens = sampling_cfg.get('use_min_new_tokens', False)
        min_new_tokens = sampling_cfg.get('min_new_tokens', 10)
        gen_max_len = model_cfg.get('max_length', 512)

        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            for task in batch:
                # Run one episode
                state = self.environment.reset(task)
                done = False
                episode_reward = 0.0
                actions_taken = []

                # Log task start for first 5 tasks
                if total < 5:
                    print(f"\n{'='*80}")
                    print(f"TASK #{total + 1}: {task.get('instruction', task.get('prompt', 'N/A'))[:100]}")
                    print(f"{'='*80}")

                for turn in range(self.config['environment']['max_turns']):
                    # 1) Render state -> text observation
                    if hasattr(self.environment, "render_text"):
                        obs_text = self.environment.render_text(state)
                    else:
                        obs_text = str(state)

                    # 2) Generate raw action text
                    with torch.no_grad():
                        gen_kwargs = dict(
                            max_new_tokens=max(16, min_new_tokens if use_min_new_tokens else 32),
                            temperature=0.7,
                            do_sample=True,
                        )

                        action_raw = self.policy.generate([obs_text], **gen_kwargs)[0]

                    # 3) Parse action (use fixed parser, NOT sanitizer!)
                    # FIXED: Sanitizer was destroying valid actions, replacing them with garbage
                    fallback_query = task.get("instruction", task.get("prompt", ""))
                    from ragen.action_parser import parse_action_from_output
                    action = parse_action_from_output(action_raw, fallback_query)

                    # Log detailed info for first 5 tasks
                    if total < 5:
                        print(f"\n  Turn {turn + 1}:")
                        print(f"    Observation: {obs_text[:150]}")
                        print(f"    Model Output (raw): '{action_raw}'")
                        print(f"    Parsed Action: '{action}'")

                    # 4) Step the environment
                    next_state, reward, done, info = self.environment.step(action)
                    episode_reward += reward
                    actions_taken.append(action)

                    # Log environment response for first 5 tasks
                    if total < 5:
                        print(f"    Env Response:")
                        print(f"      - Reward: {reward}")
                        print(f"      - Done: {done}")
                        print(f"      - Info: {info}")

                    state = next_state

                    if done:
                        break

                # Check success
                success = episode_reward > 0.5
                if success:
                    successes += 1

                # Log episode summary for first 5 tasks
                if total < 5:
                    print(f"\n  EPISODE SUMMARY:")
                    print(f"    Total Reward: {episode_reward}")
                    print(f"    Success: {'✓ YES' if success else '✗ NO'}")
                    print(f"    Actions Taken: {actions_taken}")
                    print(f"  {'='*80}")

                total += 1

        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*80}")
        success_rate = successes / max(total, 1)
        print(f"✓ Final Results: {successes}/{total} = {success_rate:.2%}")

        return success_rate

    def reinitialize_environment(self, new_product_count: int):
        """
        Reinitialize environment with new product count for curriculum learning.

        Args:
            new_product_count: New number of products for WebShop
        """
        # Update config
        self.config['environment']['num_products'] = new_product_count

        # Reinitialize environment
        if self.env_type == 'simple':
            from ragen.environments.simple_webshop import SimpleWebShopEnvironment
            self.environment = SimpleWebShopEnvironment(self.config)
        elif self.env_type == 'medium':
            from ragen.environments.medium_webshop import MediumWebShopEnvironment
            self.environment = MediumWebShopEnvironment(self.config)
        else:
            self.environment = WebShopEnvironment(self.config)

        # Update APO trainer's environment reference
        self.apo_trainer.environment = self.environment

        self.log(f"✓ Environment reinitialized with {new_product_count} products")

    def check_curriculum_progress(self, success_rate: float):
        """
        Check if curriculum should progress to next stage.

        Args:
            success_rate: Current evaluation success rate
        """
        # Record success rate
        self.curriculum.record_success_rate(success_rate)

        # Check if we should increase difficulty
        if self.curriculum.should_increase_difficulty(self.global_step):
            new_products = self.curriculum.increase_difficulty()
            self.reinitialize_environment(new_products)

    def train(self):
        """Main training loop"""
        self.log("Starting RAGEN training...")
        self.log(f"Config: {self.config}")

        # BEHAVIOR CLONING WARM-START **RE-ENABLED WITH REAL DEMOS**
        # NOW USING: ragen/real_expert_demos.py
        # Generated from actual WebShop ASIN-instruction pairs
        # All demos use REAL ASINs correctly paired with their instructions
        self.log("\n✓ Starting BC warm-start with REAL expert demonstrations...")
        self.apo_trainer.behavior_cloning_warmstart(num_steps=30)

        # Run initial evaluation (SKIP in debug mode only if requested)
        if not self.skip_initial_eval:
            self.log("Running initial evaluation...")
            initial_success = self.evaluate()
            self.log(f"Initial success rate: {initial_success:.2%}")
        else:
            self.log("⚡ Skipping initial evaluation (debug/flag)")
            initial_success = 0.0
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.log(f"\n{'='*60}")
            self.log(f"Starting Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            self.log(f"{'='*60}")
            
            self.train_epoch(epoch)
            
            # Check if we've reached max steps
            if self.global_step >= self.config['training']['max_steps']:
                self.log("Reached maximum training steps")
                break
        
        # Final evaluation (always run, even in debug)
        self.log("\n" + "="*60)
        self.log("Running final evaluation...")
        final_success = self.evaluate()
        
        self.log("\n" + "="*60)
        self.log("TRAINING COMPLETE!")
        self.log(f"Initial success rate: {initial_success:.2%}")
        self.log(f"Final success rate: {final_success:.2%}")
        self.log(f"Best success rate: {self.best_success_rate:.2%}")
        
        if not self.skip_initial_eval:
            self.log(f"Improvement: {(final_success - initial_success):+.2%}")
        
        self.log("="*60)
        
        # Save final model
        save_checkpoint(
            self.apo_trainer.policy.model,
            self.apo_trainer.optimizer,
            self.global_step,
            self.output_dir / 'final_model.pt',
            success_rate=final_success
        )
        
        # Save final metrics
        self.save_metrics()
        
        return final_success


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Debug mode - ULTRA FAST settings for local testing
    if args.debug:
        print(" Running in DEBUG mode - ULTRA FAST settings")
        config['training']['max_steps'] = 10           # Just 10 steps!
        config['training']['eval_every'] = 999         # Don't eval during training
        config['training']['save_every'] = 999         # Don't save checkpoints
        config['data']['train_size'] = 4               # Tiny dataset
        config['data']['eval_size'] = 2                # Tiny eval set
        config['environment']['max_turns'] = 2         # Only 2 turns per episode
        config['apo']['v_star_samples'] = 1            # Single V* sample
        config['apo']['adaptive_vstar'] = False        # Disable adaptive
        config['model']['max_length'] = 256            # Shorter sequences
        config['model']['sft_max_length'] = 512        # Shorter sequences
        config['environment']['num_products'] = 100
    
    # Create trainer
    trainer = RAGENTrainer(config, args.output_dir)
    
    # Set skip initial eval flag (respect CLI flags; do NOT override later)
    trainer.skip_initial_eval = args.skip_initial_eval or args.debug
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.global_step = load_checkpoint(
            trainer.apo_trainer.policy.model,
            trainer.apo_trainer.optimizer,
            args.resume
        )
        print(f"Resumed from step {trainer.global_step}")
    
    # Run evaluation only if specified
    if args.eval_only:
        print("Running evaluation only...")
        success = trainer.evaluate()
        print(f"Success Rate: {success:.2%}")
        return
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
