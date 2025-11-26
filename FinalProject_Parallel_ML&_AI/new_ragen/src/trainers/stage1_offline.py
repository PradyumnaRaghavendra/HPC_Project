"""
A*-PO Stage 1: Offline Optimal Value Estimation
Generate multiple trajectories per instruction to find V*(x)
"""
import torch
from tqdm import tqdm
from typing import List
import sys
from pathlib import Path
import numpy as np # <-- Added numpy

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.value_cache import OptimalValueCache

class Stage1OfflineTrainer:
    """
    Stage 1 of A*-PO: Estimate optimal values offline.
    """
    
    def __init__(
        self,
        env,
        policy,
        value_cache: OptimalValueCache,
        k_samples: int = 64,
        max_steps: int = 20,
        temperature: float = 1.0
    ):
        """
        Args:
            env: WebShop environment
            policy: Reference policy for sampling
            value_cache: Cache to store V* values
            k_samples: Number of trajectories per instruction
            max_steps: Max steps per trajectory
            temperature: Sampling temperature
        """
        self.env = env
        self.policy = policy
        self.value_cache = value_cache
        self.k_samples = k_samples
        self.max_steps = max_steps
        self.temperature = temperature
    
    def collect_optimal_values(
        self,
        num_instructions: int,
        verbose: bool = True
    ):
        """
        Collect V* for multiple instructions.
        
        Args:
            num_instructions: Number of unique instructions to sample
            verbose: Print progress
        """
        print(f"\n{'='*60}")
        print(f"Stage 1: Offline Value Estimation")
        print(f"{'='*60}")
        print(f"Collecting {self.k_samples} trajectories per instruction")
        print(f"Target instructions: {num_instructions}")
        print(f"{'='*60}\n")
        
        instruction_count = 0
        instructions_seen = set()
        
        pbar = tqdm(total=num_instructions, desc="Instructions processed")
        
        while instruction_count < num_instructions:
            # Reset environment to get new instruction
            obs, info = self.env.reset()
            instruction = info.get("instruction", "")
            
            if not instruction: # Skip if no instruction
                continue

            # Skip if we've already processed this instruction
            if instruction in instructions_seen:
                continue
            
            instructions_seen.add(instruction)
            
            # Collect k trajectories for this instruction
            rewards = []
            for k in range(self.k_samples):

                # Reset env for each sample to get diverse trajectories
                # For k=0, use the already-reset observation
                # For k>0, reset with the same instruction
                if k == 0:
                    obs_k = obs
                else:
                    obs_k, _ = self.env.reset_with_instruction(instruction)

                reward = self._run_single_trajectory(instruction, obs_k)
                rewards.append(reward)
            
            # Update V* with best reward
            v_star = max(rewards)
            self.value_cache.update(instruction, v_star)
            
            if verbose and (instruction_count + 1) % 10 == 0:
                print(f"\nInstruction: {instruction[:50]}...")
                print(f"  Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}")
                print(f"  V*: {v_star:.3f}")
            
            instruction_count += 1
            pbar.update(1)
        
        pbar.close()
        
        # Save cache
        self.value_cache.save()
        
        # Print statistics
        stats = self.value_cache.get_statistics()
        print(f"\n{'='*60}")
        print(f"Stage 1 Complete!")
        print(f"{'='*60}")
        print(f"Instructions processed: {stats['num_instructions']}")
        print(f"Mean V*: {stats['mean_v_star']:.3f}")
        print(f"Max V*: {stats['max_v_star']:.3f}")
        print(f"Min V*: {stats['min_v_star']:.3f}")
        print(f"{'='*60}\n")
    
    def _run_single_trajectory(
        self,
        instruction: str,
        initial_obs: str
    ) -> float:
        """
        Run single trajectory and return final TOTAL SHAPED reward.
        
        Args:
            instruction: Task instruction
            initial_obs: Initial observation
            
        Returns:
            Final total shaped reward
        """
        obs = initial_obs
        
        # <-- FIX 2: Use total SHAPED reward -->
        total_shaped_reward = 0.0
        # <-- END FIX 2 -->
        
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Generate action
            action, _ = self.policy.generate_action(
                instruction=instruction,
                observation=obs,
                temperature=self.temperature,
                sample=True
            )
            
            # Execute action
            # Assumes env.step() returns (obs, shaped_reward, done, info)
            obs, shaped_reward, done, info = self.env.step(action)
            
            # <-- FIX 2: Accumulate shaped reward -->
            total_shaped_reward += shaped_reward
            # <-- END FIX 2 -->
            
            step += 1
        
        # <-- FIX 2: Return total shaped reward -->
        return total_shaped_reward