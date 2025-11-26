"""
Multi-turn agent trainer - extends APOTrainer for RAGEN
DOES NOT modify original TinyZero code!
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from ragen.action_parser import parse_action_from_output

# Import YOUR working TinyZero trainer
from tinyzero.apo_trainer import APOTrainer
from tinyzero.rewards import compute_reward
from ragen.uncertainty_filtering import UncertaintyFilter

class _SimpleVStarCache:
    """Lightweight in-memory cache keyed by (prompt, apo-signature)."""
    def __init__(self):
        self._store = {}
    def _key(self, prompt, config):
        apo = config.get('apo', {})
        sig = (
            apo.get('beta', 0.5),
            apo.get('v_star_samples', 4),
            apo.get('adaptive_vstar', False),
        )
        return (prompt, sig)
    def get(self, prompt, config):
        return self._store.get(self._key(prompt, config))
    def save(self, prompt, config, data):
        self._store[self._key(prompt, config)] = data


class MultiTurnAPOTrainer(APOTrainer):
    """
    Multi-turn extension of APOTrainer.

    TinyZero:  prompt → single response → reward
    RAGEN:     prompt → [turn1, turn2, ..., turnN] → trajectory_reward
    """

    def __init__(self, policy_model, reference_model, config: Dict, environment):
        # Pull in all stable fixes from parent (weights, KL, masking, etc.)
        super().__init__(policy_model, reference_model, config)

        # Environment
        self.environment = environment
        self.max_turns = config.get('environment', {}).get('max_turns', 10)

        # Safe defaults for multi-turn extras
        apo_cfg = config.get('apo', {})
        self.weighting_scheme = apo_cfg.get('weighting_scheme', 'exp')  # 'exp' | 'adv' | 'shifted_advantage'
        self.adaptive_vstar = apo_cfg.get('adaptive_vstar', False)
        self.clip_grad_norm = apo_cfg.get('clip_grad_norm', 0.5)

        # Optional adaptive gradient clipping (OFF by default)
        self.adaptive_clip = apo_cfg.get('adaptive_clip', False)
        self._grad_hist = deque(maxlen=100)

        # Some generators don't support min_new_tokens; guard it
        samp_cfg = config.get('sampling', {})
        self.use_min_new_tokens = samp_cfg.get('use_min_new_tokens', False)
        self.min_new_tokens = samp_cfg.get('min_new_tokens', 10)

        # Uncertainty filtering (StarPO-S)
        self.uncertainty_filter = UncertaintyFilter(keep_percent=0.50)
        print("✓ Uncertainty filtering enabled (StarPO-S)")

        # Simple in-memory V* cache
        self.vstar_cache = _SimpleVStarCache()

        print(f"✓ Multi-turn trainer initialized (max_turns={self.max_turns})")

    def behavior_cloning_warmstart(self, num_steps=100):
        """
        Warm-start with behavior cloning on expert demonstrations.
        This teaches the model the action format before RL training.

        RAGEN paper approach: Bootstrap with demonstrations then do RL.

        Increased from 20 → 100 steps for stronger behavior cloning.
        With 50 expert demos, this provides 100 training iterations
        to deeply embed the search→click→buy pattern.
        """
        from ragen.expert_demos import format_demos_for_sft

        print(f"\n{'='*80}")
        print(f"🎓 BEHAVIOR CLONING WARM-START ({num_steps} steps)")
        print(f"{'='*80}\n")

        # Get expert demonstrations
        sft_data = format_demos_for_sft()
        print(f"✓ Loaded {len(sft_data)} expert demonstration examples")

        # SFT training loop
        self.policy.train()
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate * 5)  # Higher LR for SFT

        for step in range(num_steps):
            # Sample batch of demonstrations
            import random
            batch = random.sample(sft_data, min(4, len(sft_data)))

            total_loss = 0.0
            for prompt, target_action in batch:
                # Create full sequence: prompt + target
                full_text = prompt + target_action

                # Tokenize
                inputs = self.policy.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)

                # Move to device
                if hasattr(inputs, 'to'):
                    inputs = {k: v.to(self.policy.model.device) for k, v in inputs.items()}

                # Create labels: copy input_ids but mask prompt tokens with -100
                prompt_length = len(self.policy.tokenizer(prompt, return_tensors="pt")['input_ids'][0])
                labels = inputs['input_ids'].clone()
                labels[:, :prompt_length] = -100  # Don't compute loss on prompt

                # Forward pass - compute loss only on target action
                outputs = self.policy.model(**inputs, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()

                # Backward
                loss.backward()

            # Optimization step
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if step % 5 == 0:
                avg_loss = total_loss / len(batch)
                print(f"  SFT Step {step}/{num_steps}: Loss={avg_loss:.4f}")

        print(f"\n✓ Behavior cloning complete! Model learned action format.\n")
        self.policy.eval()

    def inject_expert_demos(self, batch: List[Dict], ratio: float = 0.2) -> List[Dict]:
        """
        BC Regularization: Replace {ratio} of batch with expert demonstrations.

        This prevents the policy from drifting too far from expert behavior
        during RL training. Following RAGEN paper's approach of mixing
        expert data throughout training.

        Args:
            batch: Current training batch from data loader
            ratio: Fraction of batch to replace with expert demos (default 0.2 = 20%)

        Returns:
            Modified batch with expert demos injected
        """
        import random
        from ragen.expert_demos import get_expert_demos

        if not batch or ratio <= 0:
            return batch

        # Calculate how many items to replace
        num_to_replace = max(1, int(len(batch) * ratio))

        # Get expert demonstrations
        expert_demos = get_expert_demos()
        if not expert_demos:
            return batch

        # Sample random expert demos
        selected_experts = random.sample(expert_demos, min(num_to_replace, len(expert_demos)))

        # Convert expert demos to batch format
        expert_batch_items = []
        for demo in selected_experts:
            expert_batch_items.append({
                'instruction': demo['instruction'],
                'prompt': demo['instruction'],
                'session': None,  # Will generate new session
                'is_expert': True  # Mark as expert for potential special handling
            })

        # Replace random items in batch with experts
        modified_batch = batch.copy()
        replace_indices = random.sample(range(len(batch)), num_to_replace)

        for i, expert_item in zip(replace_indices, expert_batch_items):
            modified_batch[i] = expert_item

        return modified_batch

    # ---------- small robustness helpers ----------

    def _safe_exp_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Stable exp((r - V*)/beta) with clipping/capping.
        Returns weights normalized to mean=1.
        """
        beta = self.beta + 1e-8
        x = (advantages / beta).clamp_(-20.0, 20.0)  # prevent overflow
        w = torch.exp(x)
        w = torch.clamp(w, max=50.0)                # avoid a few samples dominating
        return w / (w.mean().clamp_min(1e-6))

    def _grad_global_norm(self) -> float:
        total_sq = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                g = p.grad.data
                total_sq += float(g.norm(2).item() ** 2)
        return float(total_sq ** 0.5)

    def _maybe_adapt_clip(self, unclipped_norm: float):
        """
        Optionally adapt clip_grad_norm using recent gradient norms.
        Sets clip to ~2x median of recent norms.
        """
        if not self.adaptive_clip:
            return
        self._grad_hist.append(unclipped_norm)
        # Update every 10 steps after we have a decent buffer
        if self.step >= 100 and (self.step % 10 == 0) and len(self._grad_hist) >= 20:
            med = float(np.median(self._grad_hist))
            new_clip = max(0.1, 2.0 * med)
            # Smooth changes a bit
            self.clip_grad_norm = 0.8 * self.clip_grad_norm + 0.2 * new_clip
            print(f"  ↻ Adaptive clip_grad_norm → {self.clip_grad_norm:.3f} (median={med:.3f})")

    # ---------- token-level KL on completion tokens only ----------

    def _compute_kl_per_example(self, pi_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        KL(pi || ref) computed on completion tokens only. Returns per-example KL.
        pi_logits/ref_logits: [B, T, V]
        labels: [B, T] with -100 mask for prompt tokens
        """
        with torch.no_grad():
            token_mask = (labels != -100).float()  # [B, T]
        logp_pi = F.log_softmax(pi_logits, dim=-1)
        logp_ref = F.log_softmax(ref_logits, dim=-1)
        gather_idx = labels.clone()
        gather_idx[gather_idx == -100] = 0  # safe index
        lp_pi = logp_pi.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)   # [B, T]
        lp_ref = logp_ref.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) # [B, T]
        kl_tokens = (lp_pi - lp_ref) * token_mask                           # [B, T]
        per_ex_kl = (kl_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0))  # [B]
        return per_ex_kl

    # ---------- state→text adapter ----------

    def _state_to_text(self, state) -> str:
        """Convert env state to a text prompt for the policy."""
        if hasattr(self.environment, "render_text"):
            return self.environment.render_text(state)
        return str(state)

    # ---------- multi-turn rollout ----------

    def _rollout_trajectory(self, prompt: str, task_data: Dict, max_turns: Optional[int] = None) -> Optional[Dict]:
        """
        Collect a multi-turn trajectory with robust error handling.
        Returns None if rollout completely fails.
        """
        if max_turns is None:
            max_turns = self.max_turns

        try:
            # Reset env with task
            task = dict(task_data)  # avoid mutating caller
            task['prompt'] = prompt
            state = self.environment.reset(task)
            
            # Validate initial state
            if state is None or not state:
                print(f"⚠️  Invalid initial state from reset: {state}")
                return None

            actions, rewards, states = [], [], [state]
            done = False

            for turn in range(max_turns):
                try:
                    obs_text = self._state_to_text(state)
                    
                    # Validate observation text
                    if not obs_text or not isinstance(obs_text, str):
                        print(f"  Invalid observation at turn {turn}: {obs_text}")
                        break


                    # ============================================
                    # 🔍 DEBUG: PRINT FULL PROMPT (ADD THIS!)
                    # ============================================
                    if self.step <= 3 and turn == 0:
                        print(f"\n{'='*60}")
                        print(f"🔍 STEP {self.step} - FULL PROMPT TO MODEL")
                        print(f"{'='*60}")
                        print(obs_text)
                        print(f"{'='*60}\n")
                    
                    gen_kwargs = dict(
                        max_new_tokens=50,  # FIXED: was max_length=50 which truncated PROMPT!
                        # min_new_tokens=5,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=1.2,
                        pad_token_id=self.policy.tokenizer.pad_token_id,
                        eos_token_id=self.policy.tokenizer.eos_token_id
                    )
                    # if self.use_min_new_tokens:
                    #     gen_kwargs["min_new_tokens"] = self.min_new_tokens

                    # Generate action
                    # Generate action WITH system prompt + FEW-SHOT EXAMPLES
                    # Teach model the action format through examples!
                    system_prompt = """You are a WebShop agent. Your goal is to find and buy the right product.

VALID ACTIONS:
1. search[query] - search for products
2. click[B0XXXXXXXX] - click a product ID (MUST start with B0)
3. buy now - purchase the current product
4. back - go back to search

IMPORTANT: When you see product IDs like "B08X2FSR21 [SEP] Product Name [SEP] $29.99",
you MUST click the ID (B08X2FSR21), NOT the name or price!

Example 1:
Observation: Page 1 [SEP] B08X2FSR21 [SEP] Blue Headphones [SEP] $29.99
Action: click[B08X2FSR21]

Example 2:
Observation: Product: Blue Headphones, Price: $29.99, matches your search
Action: buy now

Example 3:
Observation: Back to Search [SEP] Page 1 [SEP] Next >
Action: search[specific product]

Output ONLY ONE action. NO explanations."""

                    # Combine system prompt + observation
                    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n"

                    generated = self.policy.generate([full_prompt], **gen_kwargs)
                    if not generated or not generated[0]:
                        print(f"  Empty generation at turn {turn}")
                        break

                    raw_action = generated[0]
                    # Parse to extract valid action
                    action = parse_action_from_output(raw_action, task.get('instruction', ''))

                    # Debug logging
                    if self.step <= 5 and turn == 0:
                        print(f"     RAW: '{raw_action[:60]}'")
                        print(f"     PARSED: '{action}'")

                    # ============================================
                    # 🔧 OPTION 1: DISABLE SANITIZER (RECOMMENDED)
                    # ============================================
                    # Let model see real failures - critical for learning!
                    
                    # Check if action is valid format (STRICT!)
                    action_stripped = action.strip().lower()
                    is_valid_action = (
                        (action_stripped.startswith('search[') and action_stripped.endswith(']')) or
                        (action_stripped.startswith('click[') and action_stripped.endswith(']')) or
                        action_stripped in ['buy now', 'buy', 'back']
                    )
                    
                    # ============================================
                    # 🔧 OPTION 2: USE SANITIZER (NOT RECOMMENDED)
                    # ============================================
                    # Uncomment these 3 lines if you want sanitizer:
                    # from ragen.action_sanitizer import sanitize
                    # fallback_query = task.get("instruction", task.get("prompt", ""))
                    # action = sanitize(action, fallback_query)

                    # ============================================
                    # DEBUG LOGGING - BEFORE environment step
                    # ============================================
                    if self.step <= 10 and turn == 0:
                        print(f"\n    Step {self.step}, Turn {turn}:")
                        print(f"     Task: {task.get('instruction', 'unknown')[:60]}")
                        print(f"     Observation: {obs_text[:80]}...")
                        print(f"     Raw output: '{generated[0][:60]}'")
                        print(f"     Final action: '{action[:60]}'")
                        print(f"     Valid format: {is_valid_action}")
                    
                    # ============================================
                    # EXECUTE STEP IN ENVIRONMENT
                    # ============================================
                    next_state, reward, done, info = self.environment.step(action)

                    # ============================================
                    # DEBUG LOGGING - AFTER environment step
                    # ============================================
                    if self.step <= 10 and turn == 0:
                        print(f"     Env reward: {reward:.3f}")
                        print(f"     Done: {done}")
                        print(f"     Success: {info.get('success', False)}")
                    
                    # Validate step results
                    if next_state is None:
                        print(f"  None state returned at turn {turn}")
                        next_state = info.get('error', 'Error occurred')
                    
                    actions.append(action)
                    rewards.append(float(reward))
                    states.append(next_state)
                    state = next_state
                    
                    if done:
                        break
                        
                except Exception as e:
                    print(f"⚠️  Error during turn {turn}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # Validate trajectory before returning
            if not actions:
                print(f"  No actions collected in trajectory")
                return None
            
            total_reward = float(np.sum(rewards)) if rewards else 0.0
            full_text = self._concat_trajectory(actions)
            
            # Ensure we have valid text
            if not full_text:
                print(f"  Empty full_text after concatenation")
                full_text = " "  # Minimum valid text

            # Summary logging for first 10 steps
            if self.step <= 10:
                print(f"   📊 Trajectory Summary:")
                print(f"     Total turns: {len(actions)}")
                print(f"     Actions: {[a[:30] + '...' if len(a) > 30 else a for a in actions[:3]]}")
                print(f"     Rewards: {rewards}")
                print(f"     Total reward: {total_reward:.3f}")
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'total_reward': total_reward,
                'full_text': full_text,
                'num_turns': len(actions),
                'done': done
            }
            
        except Exception as e:
            print(f"  Rollout completely failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _concat_trajectory(self, actions: List[str]) -> str:
        """
        Concatenate actions into a single completion string.
        (Prompt will be provided separately and masked.)
        """
        if not actions:
            return ""
        return "\n".join(a.strip() for a in actions if a is not None)

    # ---------- V* for multi-turn ----------

    def compute_V_star_multiturn(self, prompts: List[str], problems: List[Dict]) -> np.ndarray:
        # Early warm-up with actual rollouts (cheap but informative)
        if self.step < 10:
            print(f"  ⚡ Early V* warm-up (step {self.step}) - using 1 sample...")
            # Do 1 quick rollout per prompt to get real baseline
            V_star_values = []
            for p, prob in zip(prompts, problems):
                try:
                    traj = self._rollout_trajectory(p, prob, max_turns=3)  # Short rollouts
                    if traj is not None:
                        V_star_values.append(max(0.05, traj['total_reward']))
                    else:
                        V_star_values.append(0.05)
                except:
                    V_star_values.append(0.05)
            return np.array(V_star_values, dtype=np.float32)

        V_star_values = []
        cache_hits = 0

        # Adaptive sampling
        if self.adaptive_vstar:
            if self.step < 30:
                num_samples = 2
            elif self.step < 70:
                num_samples = 1
            else:
                num_samples = 1
        else:
            num_samples = min(self.v_star_samples, 2)

        print(f"  Computing V* for {len(prompts)} prompts ({num_samples} samples each)...")

        # Decide which prompts need compute
        to_compute, indices = [], []
        for i, p in enumerate(prompts):
            cached = self.vstar_cache.get(p, self.config)
            if cached is not None:
                V_star_values.append(cached['v_star'])
                cache_hits += 1
            else:
                V_star_values.append(None)
                to_compute.append(p)
                indices.append(i)

        # Compute for uncached prompts
        for p, i in zip(to_compute, indices):
            problem = problems[i]
            sample_rewards = []
            for _ in range(num_samples):
                try:
                    traj = self._rollout_trajectory(p, problem, max_turns=min(self.max_turns, 5))
                    if traj is not None:
                        sample_rewards.append(traj['total_reward'])
                except Exception as e:
                    print(f"    Warning: V* sampling failed: {e}")
            if not sample_rewards:
                V_star = 0.3  # conservative fallback
            else:
                rewards = np.array(sample_rewards, dtype=np.float32)
                if self.beta > 0:
                    max_r = rewards.max()
                    # numeric stability for exponent
                    exp_terms = np.exp(np.clip((rewards - max_r) / self.beta, -20.0, 20.0))
                    V_star = float(max_r + self.beta * np.log(np.mean(exp_terms)))
                else:
                    V_star = float(rewards.max())

            self.vstar_cache.save(p, self.config, {
                'v_star': V_star,
                'rewards': sample_rewards,
                'num_samples': len(sample_rewards),
            })
            V_star_values[i] = V_star

        if cache_hits:
            hit_rate = 100.0 * cache_hits / len(prompts)
            print(f"  ✓ V* cache: {cache_hits}/{len(prompts)} hits ({hit_rate:.1f}%)")

        return np.array(V_star_values, dtype=np.float32)

    # ---------- training ----------

    def train_step_multiturn(self, batch: List[Dict]) -> tuple:
        """
        Multi-turn training step - FIXED to not train on dummy data.
        """
        prompts = [item.get('instruction', item.get('prompt', '')) for item in batch]
        device = self.policy.model.device

        try:
            self.policy.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1) Collect trajectories - ONLY keep successful ones
            trajectories = []
            successful_indices = []
            rollout_failures = 0
            
            for i, (p, prob) in enumerate(zip(prompts, batch)):
                traj = self._rollout_trajectory(p, prob, self.max_turns)
                if traj is not None:
                    trajectories.append(traj)
                    successful_indices.append(i)
                else:
                    rollout_failures += 1

            # 1.5) Apply uncertainty filtering (StarPO-S)
            if self.uncertainty_filter.should_filter(self.step):
                successful_batch = [batch[i] for i in successful_indices]
                filtered_batch, trajectories, filter_stats = \
                    self.uncertainty_filter.filter_batch(successful_batch, trajectories)
                
                # Update indices after filtering
                successful_indices = list(range(len(filtered_batch)))
                batch = filtered_batch
                prompts = [item.get('instruction', item.get('prompt', '')) for item in batch]
                
                if self.step % 10 == 0 and filter_stats:
                    kept = filter_stats.get('kept', 0)
                    filtered = filter_stats.get('filtered', 0)
                    keep_ratio = filter_stats.get('keep_ratio', 1.0)
                    print(f"  🔍 Filtered: kept {kept}/{kept + filtered} "
                          f"({keep_ratio*100:.0f}%) high-variance prompts")
            
            # Skip step if too many failures
            if len(trajectories) < max(1, len(batch) * 0.3):
                print(f"  ⚠️  Too many failures ({rollout_failures}/{len(batch)}) - skipping")
                return 0.0, {
                    'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                    'avg_v_star': 0.0, 'avg_kl_penalty': 0.0,
                    'avg_turns': 0.0, 'success_rate': 0.0,
                    'weight_mean': 1.0, 'weight_std': 0.0,
                    'grad_global_norm_unclipped': 0.0,
                    'rollout_failures': rollout_failures,
                }
            
            if rollout_failures > 0:
                print(f"  ⚠️  {rollout_failures}/{len(batch)} rollouts failed")
            
            # 2) Align everything to successful rollouts
            prompts = [prompts[i] for i in successful_indices]
            batch = [batch[i] for i in successful_indices]
            
            # 3) Compute V* only for successful prompts
            V_star_np = self.compute_V_star_multiturn(prompts, problems=batch)
            V_star_t = torch.tensor(V_star_np, dtype=torch.float32, device=device)

            generated_texts = [t['full_text'] for t in trajectories]

            # 4) Compute rewards
            computed_rewards = []
            for traj, prob in zip(trajectories, batch):
                reward = compute_reward(traj, prob, step=self.step)
                computed_rewards.append(reward)

            rewards_t = torch.tensor(computed_rewards, dtype=torch.float32, device=device)

            if self.step % 5 == 0:
                print(f"   Rewards: {computed_rewards}")

            # 5) Advantages & weights
            advantages = rewards_t - V_star_t
            if len(advantages) > 1:
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            else:
                adv_norm = advantages - advantages.mean()
            adv_norm = adv_norm.clamp(-self.adv_clip, self.adv_clip).detach()

            if self.weighting_scheme == 'exp':
                weights = self._safe_exp_weights(advantages).detach()
            elif self.weighting_scheme == 'shifted_advantage':
                weights = (adv_norm + self.adv_clip).detach()
                weights = weights / (weights.mean().clamp_min(1e-6))
            else:
                weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0).detach()
                weights = weights / (weights.mean().clamp_min(1e-6))

            # 6) Tokenization
            enc_prompts = self.policy.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.sft_max_length
            )
            enc_comps = self.policy.tokenizer(
                generated_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.sft_max_length
            )
            enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
            enc_comps = {k: v.to(device) for k, v in enc_comps.items()}

            pad_id = self.policy.tokenizer.pad_token_id or self.policy.tokenizer.eos_token_id

            input_ids, attention_mask, labels = self._build_concat_with_labels(
                enc_prompts["input_ids"], enc_comps["input_ids"], pad_id
            )
            input_ids = input_ids[:, :self.sft_max_length]
            attention_mask = attention_mask[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # 7) Forward pass
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            per_ex_ce = self._per_example_ce_loss(outputs.logits, labels)

            # 8) KL term
            kl_term = torch.zeros_like(per_ex_ce)
            if self.kl_coef and self.kl_coef > 0.0 and hasattr(self.ref_model, "model"):
                with torch.no_grad():
                    ref_out = self.ref_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                kl_per_ex = self._compute_kl_per_example(outputs.logits, ref_out.logits, labels)
                kl_term = self.kl_coef * kl_per_ex
                del ref_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_ex_total = per_ex_ce + kl_term

            # 9) Weighted loss
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print("Warning: NaN/Inf weights")
                weights = torch.ones_like(per_ex_total)
            weights = weights.to(device)
            loss = (per_ex_total * weights).mean()

            # 10) Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            unclipped_norm = self._grad_global_norm()
            self._maybe_adapt_clip(unclipped_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 11) Metrics
            self.step += 1
            loss_value = float(loss.item())
            avg_reward = float(rewards_t.mean().item())
            avg_advantage = float(advantages.mean().item())
            avg_v_star = float(V_star_t.mean().item())
            avg_kl = float(kl_term.mean().item())
            avg_turns = float(np.mean([t['num_turns'] for t in trajectories]))
            success_rate = float(np.mean([1.0 if t['total_reward'] > 0.5 else 0.0 for t in trajectories]))

            w_mean = float(weights.mean().item())
            w_std  = float(weights.std().item())

            if self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print(
                    f"Step {self.step}: Loss={loss_value:.4f} (KL={avg_kl:.4f}), "
                    f"Reward={avg_reward:.3f}, Success={success_rate:.2%}"
                )

            stats = {
                'loss': loss_value,
                'avg_reward': avg_reward,
                'avg_advantage': avg_advantage,
                'avg_v_star': avg_v_star,
                'avg_kl_penalty': avg_kl,
                'avg_turns': avg_turns,
                'success_rate': success_rate,
                'weight_mean': w_mean,
                'weight_std': w_std,
                'grad_global_norm_unclipped': unclipped_norm,
                'rollout_failures': rollout_failures,
            }
            return loss_value, stats

        except Exception as e:
            print(f"\n--- Error in train_step_multiturn ---")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0, {
                'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_v_star': 0.0, 'avg_kl_penalty': 0.0,
                'avg_turns': 0.0, 'success_rate': 0.0,
                'weight_mean': 1.0, 'weight_std': 0.0,
                'grad_global_norm_unclipped': 0.0,
                'rollout_failures': len(batch),
            }