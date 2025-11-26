"""
Multi-Turn PPO Trainer for RAGEN
Extends PPO trainer to handle multi-turn environments (maze, WebShop, etc.)
Following RAGEN paper's PPO specifications
"""
import torch
from typing import List, Dict
from tinyzero.ppo_trainer import PPOTrainer
from tinyzero.rewards import compute_webshop_reward


class MultiTurnPPOTrainer:
    """
    Multi-turn PPO trainer that wraps PPO trainer and adds multi-turn rollout generation.
    Compatible with RAGEN's training loop.
    """

    def __init__(self, policy_model, ref_model, config: Dict, environment):
        """
        Initialize multi-turn PPO trainer.

        Args:
            policy_model: Policy model for action generation
            ref_model: Reference model (not used in PPO but kept for compatibility)
            config: Training configuration
            environment: Environment for rollout generation
        """
        self.config = config
        self.environment = environment
        self.policy = policy_model
        self.ref_model = ref_model

        # Initialize base PPO trainer
        self.ppo_trainer = PPOTrainer(policy_model, ref_model, config)

        # Delegate step counter to PPO trainer
        self.step = 0

        print("✓ Multi-Turn PPO Trainer initialized")
        print("  - Rollout generation: Multi-turn environment interaction")
        print("  - Training: PPO with trained critic")

    @property
    def optimizer(self):
        """Expose optimizer from wrapped PPO trainer for compatibility."""
        return self.ppo_trainer.optimizer

    def train_step(self, batch: List[Dict]) -> tuple:
        """
        Multi-turn PPO training step.

        Args:
            batch: List of task dictionaries with 'prompt' field

        Returns:
            (loss, stats): Loss value and statistics dictionary
        """
        # Generate multi-turn rollouts from environment
        rollout_batch = self._generate_rollouts(batch)

        if len(rollout_batch) == 0:
            print("⚠️  No valid rollouts generated, skipping step")
            return 0.0, {
                'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                'entropy': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_value': 0.0
            }

        # Use PPO trainer for optimization
        loss, stats = self.ppo_trainer.train_step(rollout_batch)

        # Update step counter
        self.step = self.ppo_trainer.step

        return loss, stats

    def _generate_rollouts(self, batch: List[Dict]) -> List[Dict]:
        """
        Generate multi-turn rollouts by interacting with environment.

        Args:
            batch: List of task dictionaries

        Returns:
            List of rollout dictionaries formatted for PPO training
        """
        rollouts = []
        max_turns = self.config.get('environment', {}).get('max_turns', 10)
        env_type = self.config.get('environment', {}).get('type', 'webshop')

        # CRITICAL FIX: System prompt with action format instructions
        system_prompt = """You are a WebShop agent. Output ONLY valid actions in the exact format shown below.

VALID ACTIONS:
- search[query] - Search for products (e.g., search[blue headphones])
- click[B0XXXXXXXX] - Click a product ID that starts with B0 or B1 (e.g., click[B08X2FSR21])
- buy now - Purchase the current product
- back - Go back to previous page

CRITICAL RULES:
1. Output ONLY the action, nothing else
2. NO explanations, NO reasoning, NO extra text
3. Product IDs MUST start with B0 or B1
4. Use square brackets [ ] for search and click actions

EXAMPLES:
Task: Find blue wireless headphones
→ search[blue wireless headphones]

Observation: B08X2FSR21 [SEP] Sony Headphones [SEP] $349.99 | B09ABC123 [SEP] Bose QuietComfort [SEP] $279.99
→ click[B08X2FSR21]

Observation: Product page for Sony Headphones - Blue, wireless, $349.99
→ buy now"""

        for task in batch:
            # Initialize environment with task
            instruction = task.get('instruction', task.get('prompt', ''))
            obs = self.environment.reset(instruction)

            # Build initial observation text
            obs_text = self.environment.render_text(obs) if hasattr(self.environment, 'render_text') else obs

            # Build prompt WITH system prompt (Qwen format)
            current_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n"

            # Save initial prompt for rollout tracking
            initial_prompt = current_prompt

            # Multi-turn interaction
            episode_reward = 0.0
            all_actions = []
            all_observations = []
            done = False

            for turn in range(max_turns):
                if done:
                    break

                # Generate action from policy
                try:
                    generated = self.policy.generate(
                        [current_prompt],
                        max_length=self.config.get('model', {}).get('max_length', 512),
                        min_new_tokens=1,  # Allow single token responses
                        max_new_tokens=30,  # Increased for longer product names
                        temperature=self.config.get('sampling', {}).get('temperature', 0.8),
                        do_sample=True,
                        top_p=self.config.get('sampling', {}).get('top_p', 0.9),
                        top_k=self.config.get('sampling', {}).get('top_k', 0),
                        repetition_penalty=1.2  # Discourage repeating same token
                    )[0]

                    all_actions.append(generated)
                    all_observations.append(obs)

                    # Execute action in environment
                    obs, reward, done, info = self.environment.step(generated)
                    episode_reward += reward

                    if done:
                        break

                    # Update prompt for next turn with new observation
                    obs_text = self.environment.render_text(obs) if hasattr(self.environment, 'render_text') else obs
                    current_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n"

                except Exception as e:
                    print(f"  ⚠️  Error during rollout generation: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # Create rollout entry for PPO training
            if len(all_actions) > 0:
                # Apply reward shaping for WebShop (dense rewards for better learning signal)
                if env_type == 'webshop':
                    # Build trajectory dict for reward shaping
                    trajectory = {
                        'actions': all_actions,
                        'observations': all_observations,
                        'total_reward': episode_reward
                    }

                    # Apply sophisticated reward shaping
                    # This provides dense rewards for:
                    # - Valid action formats
                    # - Relevant searches
                    # - Correct action sequences
                    # - Penalties for invalid/wrong actions
                    shaped_reward = compute_webshop_reward(trajectory, task)
                    final_reward = shaped_reward
                else:
                    # Use raw environment reward for other environments
                    final_reward = episode_reward

                rollout = {
                    'prompt': initial_prompt,  # Initial task prompt
                    'generated_text': all_actions[-1],  # Last action generated
                    'reward': final_reward,  # Use shaped reward for WebShop
                    'success': episode_reward > 0.5,  # Success based on env reward
                    'num_turns': len(all_actions)
                }

                rollouts.append(rollout)

        return rollouts

    def behavior_cloning_warmstart(self, num_steps: int = 30):
        """Delegate to PPO trainer's behavior cloning."""
        self.ppo_trainer.behavior_cloning_warmstart(num_steps)

    def sft_step(self, prompts: List[str], responses: List[str]) -> float:
        """Delegate SFT to PPO trainer."""
        return self.ppo_trainer.sft_step(prompts, responses)

    def inject_expert_demos(self, batch: List[Dict], ratio: float = 0.2) -> List[Dict]:
        """
        BC Regularization: Replace {ratio} of batch with expert demonstrations.

        Args:
            batch: Current training batch
            ratio: Fraction to replace with expert demos (default 0.2)

        Returns:
            Modified batch with expert demos injected
        """
        import random

        # Get expert demos
        env_type = self.config.get('environment', {}).get('type', 'webshop')

        if env_type == 'maze':
            from ragen.maze_expert_demos import get_maze_expert_demos
            expert_demos = get_maze_expert_demos()
        elif env_type == 'medium':
            from ragen.medium_expert_demos import get_medium_expert_demos
            expert_demos = get_medium_expert_demos()
        else:
            # USE REAL EXPERT DEMOS (generated from actual WebShop ASIN-instruction pairs)
            from ragen.real_expert_demos import get_real_expert_demos
            expert_demos = get_real_expert_demos()

        # Calculate how many to replace
        n_replace = max(1, int(len(batch) * ratio))
        n_replace = min(n_replace, len(expert_demos))  # Don't exceed available demos

        # Randomly sample demos
        sampled_demos = random.sample(expert_demos, n_replace)

        # Convert maze demo format to task format if needed
        converted_demos = []
        for demo in sampled_demos:
            if 'turns' in demo:
                # Maze format - extract first action as the task
                first_turn = demo['turns'][0]
                converted_demos.append({
                    'instruction': first_turn['observation'].split('\n\n')[0].replace('Task: ', ''),
                    'prompt': first_turn['observation']
                })
            else:
                converted_demos.append(demo)

        # Replace last n_replace items with expert demos
        return batch[:-n_replace] + converted_demos if n_replace < len(batch) else converted_demos

    def train_step_multiturn(self, batch: List[Dict]) -> tuple:
        """
        Multi-turn PPO training step.
        Wrapper for compatibility with train_ragen.py
        """
        return self.train_step(batch)
