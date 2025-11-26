"""
Multi-Turn APO Trainer for Sequential Environments (e.g., WebShop)

Adapts APO (Advantage-weighted Policy Optimization) for multi-turn tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


class MultiTurnAPOTrainer:
    """APO Trainer adapted for multi-turn sequential environments"""

    def __init__(self, policy_model, reference_model, config: Dict):
        self.policy = policy_model
        self.ref_model = reference_model
        self.config = config

        # APO hyperparams
        apo_cfg = config.get('apo', {})
        self.beta = apo_cfg.get('beta', 0.1)
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.gamma = apo_cfg.get('gamma', 1.0)
        self.gae_lambda = apo_cfg.get('gae_lambda', 1.0)

        # Generation params
        model_cfg = config.get('model', {})
        self.max_length = model_cfg.get('max_length', 128)
        self.sft_max_length = model_cfg.get('sft_max_length', 256)

        # Sampling
        samp = config.get('sampling', {})
        self.temperature = samp.get('temperature', 0.7)
        self.top_p = samp.get('top_p', 0.9)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )

        self.step = 0

    def rollout_episode(self, env, max_steps=10):
        """
        Generate a full trajectory by interacting with the environment.

        Returns:
            trajectory: List[Dict] with keys: prompt, action, reward
        """
        device = self.policy.model.device
        self.policy.model.eval()

        env.reset()
        task = env.instruction
        done = False
        trajectory = []
        steps = 0

        with torch.no_grad():
            while not done and steps < max_steps:
                obs = env.get_obs()
                prompt = f"Task: {task}\nObservation: {obs}\nAction:"

                # Generate action
                inputs = self.policy.tokenizer(prompt, return_tensors="pt").to(device)
                outputs = self.policy.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=self.top_p,
                    pad_token_id=self.policy.tokenizer.eos_token_id
                )
                action = self.policy.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                # Step environment
                result = env.step(action)
                if len(result) == 4:
                    _, reward, done, _ = result
                else:
                    _, reward, done = result[0], result[1], result[2]

                trajectory.append({
                    'prompt': prompt,
                    'action': action,
                    'reward': reward
                })

                steps += 1

        return trajectory

    def compute_advantages(self, rewards: List[float]):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        For simplicity, we'll use rewards directly as advantages.
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Simple advantage: just use rewards directly
        # (could be improved with value function estimation)
        advantages = rewards_t

        # Normalize advantages
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        return advantages

    def train_step(self, env, num_episodes=1):
        """
        Run episodes and train with APO.

        Args:
            env: Environment instance
            num_episodes: Number of episodes to run

        Returns:
            loss: Average loss
            stats: Training statistics
        """
        device = self.policy.model.device
        all_losses = []
        total_reward = 0.0
        total_steps = 0

        for _ in range(num_episodes):
            # Generate trajectory
            trajectory = self.rollout_episode(env)

            if len(trajectory) == 0:
                continue

            # Extract data
            prompts = [step['prompt'] for step in trajectory]
            actions = [step['action'] for step in trajectory]
            rewards = [step['reward'] for step in trajectory]

            # Compute advantages
            advantages = self.compute_advantages(rewards)
            advantages = advantages.to(device)

            # Compute weights from advantages
            # Use shifted advantages to ensure positive weights
            weights = (advantages + 3.0).clamp(min=0.1, max=10.0)

            # Tokenize for training
            prompt_enc = self.policy.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            action_enc = self.policy.tokenizer(
                actions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=False
            )

            prompt_ids = prompt_enc['input_ids'].to(device)
            action_ids = action_enc['input_ids'].to(device)

            # Concatenate prompt + action
            input_ids = torch.cat([prompt_ids, action_ids], dim=1)
            labels = input_ids.clone()
            labels[:, :prompt_ids.size(1)] = -100  # Mask prompt

            # Truncate
            input_ids = input_ids[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # Ensure correct dtype (must be Long for embedding layer)
            input_ids = input_ids.long()
            labels = labels.long()

            # Forward pass
            self.policy.model.train()
            outputs = self.policy.model(input_ids=input_ids, labels=None)
            logits = outputs.logits

            # Compute per-example loss
            per_ex_loss = self._per_example_ce_loss(logits, labels)

            # Weight by advantages
            weighted_loss = (per_ex_loss * weights).mean()

            # Backward
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            all_losses.append(weighted_loss.item())
            total_reward += sum(rewards)
            total_steps += len(trajectory)

        self.step += 1

        # Stats
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0

        stats = {
            'loss': avg_loss,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
        }

        return avg_loss, stats

    def _per_example_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean CE per example over unmasked tokens."""
        B, T, V = logits.shape

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        # Loss per token
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='none'
        ).view(B, T - 1)

        # Mask and average per example
        token_mask = (shift_labels != -100).float()
        per_ex_loss = token_losses.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        per_ex_loss = torch.nan_to_num(per_ex_loss, nan=0.0)

        return per_ex_loss
