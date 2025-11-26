"""
A*-PO Trainer for WebShop
Based on TinyZero's proven weighted SFT approach
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path


class APOTrainer:
    """
    A*-PO Trainer using Weighted Supervised Learning

    Key difference from REINFORCE:
    - Generates completions first
    - Trains with weighted cross-entropy (teacher-forcing)
    - Converts advantages to positive weights
    - More stable than policy gradient
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        env,
        config: Dict
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.env = env
        self.config = config

        # A*-PO hyperparameters
        apo_cfg = config.get('apo', {})
        self.beta = apo_cfg.get('beta', 0.5)
        self.v_star_samples = apo_cfg.get('v_star_samples', 8)
        self.learning_rate = apo_cfg.get('learning_rate', 5e-6)
        self.kl_coef = apo_cfg.get('kl_coef', 0.02)
        self.adv_clip = apo_cfg.get('adv_clip', 3.0)
        self.clip_grad_norm = apo_cfg.get('clip_grad_norm', 1.0)

        # Generation settings
        model_cfg = config.get('model', {})
        self.max_steps = model_cfg.get('max_steps', 15)
        self.max_length = model_cfg.get('max_length', 512)

        # Sampling
        samp = config.get('sampling', {})
        self.temperature = samp.get('temperature', 0.9)
        self.top_p = samp.get('top_p', 0.95)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # V* cache with adaptive sampling
        self.vstar_cache = {}
        self.adaptive_vstar = apo_cfg.get('adaptive_vstar', True)

        # Training state
        self.step = 0

        print("✓ A*-PO Trainer initialized (Weighted SFT mode)")
        print(f"  Beta: {self.beta}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  V* samples: {self.v_star_samples}")
        print(f"  KL coefficient: {self.kl_coef}")
        print(f"  Advantage clip: {self.adv_clip}")

    @torch.no_grad()
    def compute_V_star(self, instructions: List[str]) -> np.ndarray:
        """
        Compute V* with caching and adaptive sampling.

        V* = smooth max of rewards from k sampled trajectories
        V* = max_r + β * log(mean(exp((r_i - max_r) / β)))
        """
        V_star_values = []
        cache_hits = 0

        # Adaptive sampling based on training progress
        if self.adaptive_vstar:
            if self.step < 30:
                num_samples = self.v_star_samples
            elif self.step < 100:
                num_samples = max(4, self.v_star_samples // 2)
            else:
                num_samples = max(2, self.v_star_samples // 4)
        else:
            num_samples = self.v_star_samples

        print(f"  Computing V* for {len(instructions)} instructions ({num_samples} samples each)...")

        for instruction in instructions:
            # Check cache
            if instruction in self.vstar_cache:
                V_star_values.append(self.vstar_cache[instruction])
                cache_hits += 1
                continue

            # Sample trajectories with reference model
            rewards = []
            for _ in range(num_samples):
                reward = self._rollout_trajectory(
                    instruction,
                    model=self.ref_model,
                    temperature=1.0,  # Higher temp for exploration
                    for_vstar=True
                )
                rewards.append(reward)

            # Compute smooth max V*
            rewards = np.array(rewards, dtype=np.float32)
            if len(rewards) == 0 or rewards.max() == rewards.min():
                V_star = float(rewards.mean()) if len(rewards) > 0 else 0.0
            else:
                max_r = rewards.max()
                exp_terms = np.exp((rewards - max_r) / (self.beta + 1e-8))
                V_star = float(max_r + self.beta * np.log(np.mean(exp_terms) + 1e-8))

            # Cache
            self.vstar_cache[instruction] = V_star
            V_star_values.append(V_star)

        if cache_hits > 0:
            print(f"  ✓ V* cache: {cache_hits}/{len(instructions)} hits ({100*cache_hits/len(instructions):.1f}%)")

        return np.array(V_star_values, dtype=np.float32)

    def _rollout_trajectory(
        self,
        instruction: str,
        model,
        temperature: float = 0.9,
        for_vstar: bool = False
    ) -> float:
        """
        Roll out a trajectory in WebShop environment.
        Returns total reward.
        """
        obs, info = self.env.reset()

        # Keep resetting until we get the target instruction
        # (WebShop doesn't support setting instruction directly)
        current_instruction = info.get('instruction', '')
        attempts = 0
        max_attempts = 50

        while current_instruction != instruction and attempts < max_attempts:
            obs, info = self.env.reset()
            current_instruction = info.get('instruction', '')
            attempts += 1

        if current_instruction != instruction:
            # Couldn't find instruction, return 0 reward
            return 0.0

        # Generate trajectory
        actions_text = []
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < self.max_steps:
            # Generate action
            action = model.generate_action(
                instruction=instruction,
                observation=obs,
                previous_actions=actions_text,
                temperature=temperature,
                do_sample=True,
                top_p=self.top_p
            )

            actions_text.append(action)

            # Execute in environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            step += 1

        # For V* computation, use raw reward (binary success)
        if for_vstar:
            raw_reward = info.get('raw_reward', 0.0)
            return raw_reward
        else:
            # For training, use shaped reward
            return total_reward

    def _build_training_batch(
        self,
        instructions: List[str],
        trajectories: List[Dict]
    ):
        """
        Build training batch with prompt masking.

        For each trajectory:
        - Concatenate: instruction + observation + action
        - Mask instruction/observation tokens (only train on actions)
        - Return input_ids, attention_mask, labels
        """
        device = self.policy.model.device
        tokenizer = self.policy.tokenizer

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for instruction, traj in zip(instructions, trajectories):
            observations = traj['observations']
            actions = traj['actions']

            # Build full text with special tokens to mark actions
            full_text = f"Instruction: {instruction}\n\n"
            action_starts = []

            for obs, action in zip(observations, actions):
                obs_start = len(full_text)
                full_text += f"Observation: {obs}\n"
                action_start = len(full_text)
                full_text += f"Action: {action}\n"
                action_end = len(full_text)

                action_starts.append((action_start, action_end))

            # Tokenize full text
            encoded = tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask'][0]

            # Create labels: -100 for non-action tokens, input_ids for action tokens
            labels = torch.full_like(input_ids, -100)

            # Find action token spans and unmask them
            # This is approximate - we tokenize each action separately to find spans
            current_pos = 0
            decoded_so_far = ""

            for obs, action in zip(observations, actions):
                # Find where this action's tokens are
                action_text = f"Action: {action}\n"
                action_tokens = tokenizer(action_text, add_special_tokens=False)['input_ids']

                # Search for action tokens in input_ids
                # This is a simplified approach - in production, use more robust alignment
                # For now, we'll just unmask the last N tokens proportionally
                pass

            # Simplified: unmask last 30% of tokens (actions are at the end)
            # This is a heuristic - TinyZero uses more sophisticated alignment
            total_tokens = attention_mask.sum().item()
            action_ratio = 0.3
            unmask_from = int(total_tokens * (1 - action_ratio))
            labels[unmask_from:] = input_ids[unmask_from:]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        # Stack into batch
        input_ids = torch.stack(all_input_ids).to(device)
        attention_mask = torch.stack(all_attention_mask).to(device)
        labels = torch.stack(all_labels).to(device)

        return input_ids, attention_mask, labels

    def _per_example_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean CE loss per example over unmasked tokens."""
        B, T, V = logits.shape

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        # Compute loss per token
        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='none'
        ).view(B, T - 1)

        # Average over non-masked tokens per example
        token_mask = (shift_labels != -100).float()
        per_ex_loss = (token_losses * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)

        # Handle NaN
        per_ex_loss = torch.nan_to_num(per_ex_loss, nan=0.0)

        return per_ex_loss

    def _compute_kl_loss(
        self,
        logits_pi: torch.Tensor,
        logits_ref: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence per example on completion tokens."""
        logits_pi_shifted = logits_pi[..., :-1, :].contiguous()
        logits_ref_shifted = logits_ref[..., :-1, :].contiguous()
        labels_shifted = labels[..., 1:].contiguous()

        logp_pi = F.log_softmax(logits_pi_shifted, dim=-1)
        logp_ref = F.log_softmax(logits_ref_shifted, dim=-1)

        token_mask = (labels_shifted != -100).float()

        # KL divergence per token
        kl_div_tokens = F.kl_div(logp_ref, logp_pi, log_target=True, reduction='none').sum(-1)
        kl_div_tokens = kl_div_tokens * token_mask

        # Average per example
        kl_per_ex = kl_div_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        kl_per_ex = torch.nan_to_num(kl_per_ex, nan=0.0)

        return kl_per_ex

    def train_step(self, batch_instructions: List[str]) -> tuple:
        """
        Single A*-PO training step with weighted SFT.

        Steps:
        1. Compute V* for instructions
        2. Generate trajectories from policy
        3. Compute rewards and advantages
        4. Convert advantages to positive weights
        5. Train with weighted cross-entropy loss
        6. Add KL regularization
        """
        device = self.policy.model.device

        try:
            # Step 1: Compute V*
            V_star_np = self.compute_V_star(batch_instructions)
            V_star_t = torch.tensor(V_star_np, dtype=torch.float32, device=device)

            # Step 2: Generate trajectories from policy
            print(f"  Generating {len(batch_instructions)} trajectories from policy...")
            trajectories = []
            rewards = []

            for instruction in batch_instructions:
                # Roll out trajectory
                obs, info = self.env.reset()

                # Find this instruction
                attempts = 0
                while info.get('instruction', '') != instruction and attempts < 50:
                    obs, info = self.env.reset()
                    attempts += 1

                if info.get('instruction', '') != instruction:
                    # Skip this instruction
                    continue

                # Generate trajectory
                observations = []
                actions = []
                total_reward = 0.0
                done = False
                step = 0

                while not done and step < self.max_steps:
                    action = self.policy.generate_action(
                        instruction=instruction,
                        observation=obs,
                        previous_actions=actions,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=self.top_p
                    )

                    observations.append(obs)
                    actions.append(action)

                    obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    step += 1

                trajectories.append({
                    'observations': observations,
                    'actions': actions,
                    'instruction': instruction
                })
                rewards.append(total_reward)

            if len(trajectories) == 0:
                print("  ⚠️ No valid trajectories generated!")
                return 0.0, {
                    'loss': 0.0,
                    'avg_reward': 0.0,
                    'avg_advantage': 0.0,
                    'avg_v_star': 0.0,
                    'weight_mean': 0.0,
                    'weight_std': 0.0,
                    'num_trajectories': 0,
                    'kl_div': 0.0,
                    'adv_norm_mean': 0.0,
                    'adv_norm_std': 0.0
                }

            # Step 3: Compute advantages
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            # Truncate V_star to match number of successful trajectories
            V_star_t = V_star_t[:len(trajectories)]

            advantages = rewards_t - V_star_t

            # Step 4: Normalize and convert to weights
            adv_mean = advantages.mean()

            # Handle small batch sizes (need at least 2 samples for std)
            if len(advantages) > 1:
                adv_std = advantages.std() + 1e-6
                adv_norm = (advantages - adv_mean) / adv_std
            else:
                # With 1 sample, just use the raw advantage
                adv_norm = advantages - adv_mean

            adv_norm = adv_norm.clamp(-self.adv_clip, self.adv_clip)

            # Convert to positive weights (KEY DIFFERENCE from REINFORCE!)
            weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0).detach()

            # Step 5: Build training batch
            # Simplified: concatenate all actions from trajectory
            batch_texts = []
            for instruction, traj in zip(batch_instructions[:len(trajectories)], trajectories):
                # Format: Instruction + all (observation, action) pairs
                text = f"Instruction: {instruction}\n\n"
                for obs, action in zip(traj['observations'], traj['actions']):
                    text += f"Observation: {obs[:200]}\nAction: {action}\n"
                batch_texts.append(text)

            # Tokenize
            tokenizer = self.policy.tokenizer
            encoded = tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = input_ids.clone()

            # Mask padding tokens
            labels[labels == tokenizer.pad_token_id] = -100

            # Step 6: Forward pass
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            policy_logits = outputs.logits

            # Compute per-example CE loss
            per_ex_ce_loss = self._per_example_ce_loss(policy_logits, labels)

            # Step 7: KL regularization (optional)
            kl_term = torch.zeros_like(per_ex_ce_loss)
            if self.kl_coef > 0:
                with torch.no_grad():
                    ref_outputs = self.ref_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                ref_logits = ref_outputs.logits.detach()
                kl_per_ex = self._compute_kl_loss(policy_logits, ref_logits, labels)
                kl_term = self.kl_coef * kl_per_ex

            per_ex_loss = per_ex_ce_loss + kl_term

            # Step 8: Apply weights and compute final loss
            weighted_losses = per_ex_loss * weights
            loss = weighted_losses.mean()

            # Step 9: Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Metrics
            self.step += 1

            stats = {
                'loss': float(loss.item()),
                'avg_reward': float(rewards_t.mean().item()),
                'avg_advantage': float(advantages.mean().item()),
                'avg_v_star': float(V_star_t.mean().item()),
                'avg_kl_penalty': float(kl_term.mean().item()),
                'adv_norm_mean': float(adv_norm.mean().item()),
                'adv_norm_std': float(adv_norm.std().item()) if len(adv_norm) > 1 else 0.0,
                'weight_mean': float(weights.mean().item()),
                'weight_std': float(weights.std().item()) if len(weights) > 1 else 0.0,
                'num_trajectories': len(trajectories)
            }

            return float(loss.item()), stats

        except Exception as e:
            print(f"\n❌ Error in train_step: {e}")
            import traceback
            traceback.print_exc()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return 0.0, {
                'loss': 0.0,
                'avg_reward': 0.0,
                'avg_advantage': 0.0,
                'avg_v_star': 0.0,
                'avg_kl_penalty': 0.0,
                'adv_norm_mean': 0.0,
                'adv_norm_std': 0.0,
                'weight_mean': 1.0,
                'weight_std': 0.0,
                'num_trajectories': 0
            }

    def save_vstar_cache(self, path: str):
        """Save V* cache to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.vstar_cache, f, indent=2)

        print(f"✓ V* cache saved: {len(self.vstar_cache)} entries")

    def load_vstar_cache(self, path: str):
        """Load V* cache from disk."""
        path = Path(path)
        if not path.exists():
            print(f"⚠️ V* cache not found at {path}")
            return

        with open(path, 'r') as f:
            self.vstar_cache = json.load(f)

        print(f"✓ V* cache loaded: {len(self.vstar_cache)} entries")
