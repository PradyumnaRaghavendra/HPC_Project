"""
RAGEN Trainer
Main training loop for multi-turn WebShop agent
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

class RAGENTrainer:
    """
    Trains policy on WebShop using RL with dense rewards.
    """
    
    def __init__(
        self,
        policy,
        value_model,
        rollout_collector,
        learning_rate: float = 1e-5,
        value_lr: float = 3e-5,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.1,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Args:
            policy: Policy model
            value_model: Value model
            rollout_collector: Trajectory collector
            learning_rate: Policy learning rate
            value_lr: Value model learning rate
            clip_epsilon: PPO clipping parameter
            kl_coeff: KL penalty coefficient
            entropy_coeff: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.policy = policy
        self.value_model = value_model
        self.rollout_collector = rollout_collector
        
        # Hyperparameters
        self.lr = learning_rate
        self.value_lr = value_lr
        self.clip_epsilon = clip_epsilon
        self.kl_coeff = kl_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.lr
        )
        self.value_optimizer = optim.AdamW(
            self.value_model.parameters(),
            lr=self.value_lr
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
        self.best_success_rate = 0.0
    
    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 4,
        ppo_epochs: int = 3,
        eval_frequency: int = 10,
        save_frequency: int = 50,
        temperature: float = 1.0
    ):
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Rollouts per iteration
            ppo_epochs: PPO update epochs per iteration
            eval_frequency: Evaluate every N iterations
            save_frequency: Save checkpoint every N iterations
            temperature: Sampling temperature
        """
        print(f"\n{'='*60}")
        print(f"Starting RAGEN Training")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Episodes per iteration: {episodes_per_iteration}")
        print(f"PPO epochs: {ppo_epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"{'='*60}\n")
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Collect trajectories
            print(f"\nCollecting {episodes_per_iteration} trajectories...")
            trajectories = self.rollout_collector.collect_batch(
                num_episodes=episodes_per_iteration,
                temperature=temperature,
                verbose=True
            )
            
            # Log rollout statistics
            stats = self.rollout_collector.get_statistics(trajectories)
            print(f"\nRollout Statistics:")
            print(f"  Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
            print(f"  Mean Length: {stats['mean_length']:.1f}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            
            self._log_metrics(stats, iteration, "rollout")
            
            # Prepare training data
            train_data = self._prepare_training_data(trajectories)
            
            # PPO updates
            print(f"\nPerforming {ppo_epochs} PPO update epochs...")
            for epoch in range(ppo_epochs):
                metrics = self._ppo_update(train_data)
                
                if epoch == ppo_epochs - 1:  # Log last epoch
                    print(f"\nTraining Metrics (Epoch {epoch + 1}):")
                    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                    print(f"  Value Loss: {metrics['value_loss']:.4f}")
                    print(f"  Entropy: {metrics['entropy']:.4f}")
                    print(f"  KL Divergence: {metrics['kl_div']:.4f}")
                    
                    self._log_metrics(metrics, iteration, "train")
            
            self.global_step += 1
            
            # Evaluation
            if (iteration + 1) % eval_frequency == 0:
                print(f"\nRunning evaluation...")
                eval_stats = self._evaluate(num_episodes=10)
                print(f"\nEvaluation Results:")
                print(f"  Mean Reward: {eval_stats['mean_reward']:.3f}")
                print(f"  Success Rate: {eval_stats['success_rate']:.1%}")
                
                self._log_metrics(eval_stats, iteration, "eval")
                
                # Save best model
                if eval_stats['success_rate'] > self.best_success_rate:
                    self.best_success_rate = eval_stats['success_rate']
                    self._save_checkpoint(f"best_model")
                    print(f"  ✓ New best model saved! (Success rate: {self.best_success_rate:.1%})")
            
            # Periodic checkpoint
            if (iteration + 1) % save_frequency == 0:
                self._save_checkpoint(f"checkpoint_{iteration+1}")
                print(f"  ✓ Checkpoint saved")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Success Rate: {self.best_success_rate:.1%}")
        print(f"{'='*60}\n")
        
        self.writer.close()
    
    def _prepare_training_data(self, trajectories: List) -> Dict:
        """Prepare trajectories for training."""
        instructions = []
        observations = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        
        for traj in trajectories:
            # Compute returns
            traj_returns = self.rollout_collector.compute_returns(traj)
            traj_advantages = self.rollout_collector.compute_advantages(traj, traj_returns)
            
            for trans, ret, adv in zip(traj.transitions, traj_returns, traj_advantages):
                instructions.append(trans.instruction)
                observations.append(trans.observation)
                actions.append(trans.action)
                old_log_probs.append(trans.log_prob)
                returns.append(ret)
                advantages.append(adv)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            "instructions": instructions,
            "observations": observations,
            "actions": actions,
            "old_log_probs": torch.tensor(old_log_probs),
            "returns": torch.tensor(returns),
            "advantages": torch.tensor(advantages)
        }
    
    def _ppo_update(self, data: Dict) -> Dict:
        """Single PPO update."""
        # Compute new log probs
        new_log_probs = []
        entropies = []
        
        for inst, obs, action in zip(data["instructions"], data["observations"], data["actions"]):
            log_prob, entropy = self.policy.compute_log_probs(inst, obs, action)
            new_log_probs.append(log_prob)
            entropies.append(entropy)
        
        new_log_probs = torch.stack(new_log_probs)
        entropies = torch.stack(entropies)
        
        # Compute ratios
        ratio = torch.exp(new_log_probs - data["old_log_probs"])
        
        # Compute policy loss (PPO clipped objective)
        advantages = data["advantages"]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy_loss = -entropies.mean()
        
        # KL penalty
        kl_div = (data["old_log_probs"] - new_log_probs).mean()
        
        # Total policy loss
        total_policy_loss = policy_loss + self.entropy_coeff * entropy_loss + self.kl_coeff * kl_div
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        
        # Update value function
        values = self.value_model.forward_batch(data["instructions"], data["observations"])
        value_loss = nn.MSELoss()(values, data["returns"])
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
            "kl_div": kl_div.item()
        }
    
    def _evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate current policy."""
        trajectories = self.rollout_collector.collect_batch(
            num_episodes=num_episodes,
            temperature=0.7,  # Lower temperature for evaluation
            verbose=False
        )
        return self.rollout_collector.get_statistics(trajectories)
    
    def _log_metrics(self, metrics: Dict, iteration: int, prefix: str):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, iteration)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = self.checkpoint_dir / name
        save_path.mkdir(exist_ok=True)
        
        self.policy.save(str(save_path / "policy"))
        torch.save(self.value_model.state_dict(), save_path / "value_model.pt")
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_success_rate": self.best_success_rate
        }
        with open(save_path / "state.json", "w") as f:
            json.dump(state, f)