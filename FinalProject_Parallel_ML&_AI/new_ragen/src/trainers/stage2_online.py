"""
A*-PO Stage 2: Online Policy Optimization
Train policy using optimal advantages A*(x,y) = r(x,y) - V*(x)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, List # <-- Added List
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class Stage2OnlineTrainer:
    """
    Stage 2 of A*-PO: Online policy optimization.
    """
    
    def __init__(
        self,
        env,
        policy,
        value_cache,
        learning_rate: float = 5e-6,
        beta: float = 0.5,  # A*-PO advantage scaling
        batch_size: int = 8,
        max_grad_norm: float = 1.0,
        max_steps: int = 20,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Args:
            env: WebShop environment
            policy: Policy to train
            value_cache: Pre-computed V* values
            learning_rate: Learning rate (default: 5e-6)
            beta: A*-PO advantage scaling factor (default: 0.5)
            batch_size: Number of episodes per update (default: 8)
            max_grad_norm: Gradient clipping (default: 1.0)
            max_steps: Max steps per episode (default: 20)
            log_dir: Tensorboard logs
            checkpoint_dir: Model checkpoints
        """
        self.env = env
        self.policy = policy
        self.value_cache = value_cache
        self.beta = beta
        self.batch_size = batch_size # <-- FIX 3
        self.max_grad_norm = max_grad_norm
        self.max_steps = max_steps
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.global_step = 0
        self.best_reward = -float('inf') # <-- Start at -inf
    
    def train(
        self,
        num_iterations: int,
        eval_frequency: int = 10,
        save_frequency: int = 50,
        temperature: float = 1.0
    ):
        """
        Main training loop for Stage 2.
        """
        print(f"\n{'='*60}")
        print(f"Stage 2: Online Policy Optimization")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"A-PO Scale (beta): {self.beta}")
        print(f"{'='*60}\n")
        
        # <-- FIX 3: Add batch buffer -->
        batch_data = []
        
        for iteration in range(num_iterations):
            print(f"-- Iteration {iteration + 1}/{num_iterations} --")
            
            # Collect trajectory
            trajectory_data = self._collect_trajectory(temperature)
            batch_data.append(trajectory_data)
            
            # Log rollout info
            print(f"  Rollout: R_shaped={trajectory_data['shaped_reward']:.3f}, R_raw={trajectory_data['raw_reward']:.3f}, V*={trajectory_data['v_star']:.3f}, Adv={trajectory_data['advantage']:.3f}")
            
            # <-- FIX 3: Update policy only when batch is full -->
            if len(batch_data) >= self.batch_size:
                
                # Update policy
                metrics = self._update_policy(batch_data)
                
                print(f"  Training (Batch Update):")
                print(f"    Loss: {metrics['loss']:.4f}")
                print(f"    Mean Log Prob: {metrics['mean_log_prob']:.4f}")
                print(f"    Mean Advantage: {metrics['mean_advantage']:.4f}")
                
                # Log metrics
                self._log_metrics(metrics, iteration)
                
                # Clear batch
                batch_data = []
                self.global_step += 1 # Increment global step per *update*
            
            # Evaluation
            if (iteration + 1) % eval_frequency == 0:
                eval_stats = self._evaluate(num_episodes=10, temperature=0.7)
                print(f"\n  Evaluation:")
                print(f"    Mean Shaped Reward: {eval_stats['mean_shaped_reward']:.3f}")
                print(f"    Mean Raw Reward: {eval_stats['mean_raw_reward']:.3f}")
                print(f"    Success Rate: {eval_stats['success_rate']:.1%}")
                
                self._log_metrics(eval_stats, iteration, prefix="eval")
                
                # Save best model
                if eval_stats['mean_shaped_reward'] > self.best_reward:
                    self.best_reward = eval_stats['mean_shaped_reward']
                    self._save_checkpoint("best_model")
                    print(f"    ✓ New best model!")
            
            # Periodic checkpoint
            if (iteration + 1) % save_frequency == 0:
                self._save_checkpoint(f"checkpoint_{iteration+1}")
                print(f"  ✓ Checkpoint saved")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Mean Shaped Reward: {self.best_reward:.3f}")
        print(f"{'='*60}\n")
        
        self.writer.close()
    
    def _collect_trajectory(self, temperature: float) -> Dict:
        """Collect single trajectory and compute advantage."""
        obs, info = self.env.reset()
        instruction = info.get("instruction", "")
        v_star = self.value_cache.get(instruction, default=0.0)
        
        actions = []
        observations = [obs]
        done = False
        step = 0
        
        # <-- FIX 1: Use total SHAPED reward -->
        total_shaped_reward = 0.0
        final_step_info = {}
        # <-- END FIX 1 -->
        
        while not done and step < self.max_steps:
            action, _ = self.policy.generate_action(
                instruction=instruction,
                observation=obs,
                temperature=temperature,
                sample=True
            )
            
            actions.append(action)
            # Assumes env.step() returns (obs, shaped_reward, done, info)
            obs, shaped_reward, done, step_info = self.env.step(action)
            
            observations.append(obs)
            final_step_info = step_info # Store last info
            
            # <-- FIX 1: Accumulate shaped reward -->
            total_shaped_reward += shaped_reward
            # <-- END FIX 1 -->
            
            step += 1
        
        # Get raw reward for success metric
        raw_reward = final_step_info.get("raw_reward", 0.0)
        
        # <-- FIX 1: Compute advantage using total SHAPED reward -->
        advantage = total_shaped_reward - v_star
        # <-- END FIX 1 -->
        
        return {
            "instruction": instruction,
            "observations": observations[:-1],  # Exclude last obs
            "actions": actions,
            "raw_reward": raw_reward,
            "shaped_reward": total_shaped_reward,
            "v_star": v_star,
            "advantage": advantage,
            "num_steps": len(actions)
        }
    
    def _update_policy(self, batch_data: List[Dict]) -> Dict:
        """
        Update policy using A*-PO loss on a batch.

        A*-PO uses policy gradient with optimal advantages:
        Loss = -E[log π(y|x) * A*(x,y)]
        where A*(x,y) = R(x,y) - V*(x)
        """

        batch_log_probs = []
        batch_advantages = []

        for trajectory_data in batch_data:
            instruction = trajectory_data["instruction"]
            observations = trajectory_data["observations"]
            actions = trajectory_data["actions"]

            # Compute log probs for all actions in trajectory
            total_log_prob = 0.0

            for obs, action in zip(observations, actions):
                log_prob, _ = self.policy.compute_log_probs(
                    instruction=instruction,
                    observation=obs,
                    action=action,
                    requires_grad=True
                )
                total_log_prob = total_log_prob + log_prob

            batch_log_probs.append(total_log_prob)
            batch_advantages.append(trajectory_data["advantage"])

        # Stack all tensors
        log_probs_tensor = torch.stack(batch_log_probs)
        advantages_tensor = torch.tensor(batch_advantages, device=log_probs_tensor.device, dtype=torch.float32)

        # Normalize advantages for stable training
        advantages_normalized = advantages_tensor
        if len(advantages_tensor) > 1:
            advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Clip advantages to prevent extreme updates
        # This helps with stability when V* estimates are poor
        advantages_clipped = torch.clamp(advantages_normalized, -5.0, 5.0)

        # A*-PO Policy Gradient Loss
        # Maximize log_prob weighted by advantage
        # Note: beta is used as a scaling factor to control update magnitude
        policy_loss = -(log_probs_tensor * advantages_clipped).mean() * self.beta

        # Optional: Add small entropy bonus to encourage exploration
        # entropy_bonus = 0.01 * entropy.mean() if needed

        loss = policy_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_log_prob": log_probs_tensor.mean().item(),
            "mean_advantage": advantages_tensor.mean().item(),
            "std_advantage": advantages_tensor.std().item() if len(advantages_tensor) > 1 else 0.0,
            "mean_advantage_normalized": advantages_normalized.mean().item(),
            "advantages_clipped_ratio": (advantages_normalized != advantages_clipped).float().mean().item()
        }
    
    def _evaluate(self, num_episodes: int, temperature: float) -> Dict:
        """Evaluate current policy."""
        shaped_rewards = []
        raw_rewards = []
        successes = []
        
        for _ in range(num_episodes):
            traj = self._collect_trajectory(temperature)
            shaped_rewards.append(traj["shaped_reward"])
            raw_rewards.append(traj["raw_reward"])
            successes.append(traj["raw_reward"] > 0.5)
        
        return {
            "mean_shaped_reward": np.mean(shaped_rewards),
            "mean_raw_reward": np.mean(raw_rewards),
            "std_raw_reward": np.std(raw_rewards),
            "success_rate": np.mean(successes)
        }
    
    def _log_metrics(self, metrics: Dict, iteration: int, prefix: str = "train"):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, iteration)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = self.checkpoint_dir / name
        save_path.mkdir(exist_ok=True, parents=True)
        
        self.policy.save(str(save_path / "policy"))
        
        state = {
            "global_step": self.global_step,
            "best_reward": self.best_reward
        }
        with open(save_path / "state.json", "w") as f:
            json.dump(state, f)