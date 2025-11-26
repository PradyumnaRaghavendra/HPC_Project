"""
Rollout Collection for RAGEN
Collects multi-turn trajectories from WebShop
"""
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class Transition:
    """Single step transition."""
    instruction: str
    observation: str
    action: str
    reward: float
    next_observation: str
    done: bool
    log_prob: float
    value: float
    
@dataclass
class Trajectory:
    """Complete episode trajectory."""
    transitions: List[Transition]
    total_reward: float
    success: bool
    length: int
    
    def __len__(self):
        return len(self.transitions)

class RolloutCollector:
    """
    Collects trajectories from environment using current policy.
    """
    
    def __init__(
        self,
        env,
        policy,
        value_model,
        max_steps: int = 20,
        gamma: float = 0.99
    ):
        """
        Args:
            env: WebShop environment
            policy: Policy model
            value_model: Value model
            max_steps: Maximum steps per episode
            gamma: Discount factor
        """
        self.env = env
        self.policy = policy
        self.value_model = value_model
        self.max_steps = max_steps
        self.gamma = gamma
    
    def collect_trajectory(
        self,
        temperature: float = 1.0,
        verbose: bool = False
    ) -> Trajectory:
        """
        Collect single trajectory.
        
        Args:
            temperature: Sampling temperature
            verbose: Print episode info
            
        Returns:
            Complete trajectory with all transitions
        """
        # Reset environment
        obs, info = self.env.reset()
        instruction = info["instruction"]
        
        transitions = []
        total_reward = 0.0
        done = False
        step = 0
        
        if verbose:
            print(f"\n=== Episode Start ===")
            print(f"Instruction: {instruction}")
        
        while not done and step < self.max_steps:
            # Generate action
            action, action_info = self.policy.generate_action(
                instruction=instruction,
                observation=obs,
                temperature=temperature,
                sample=True
            )
            
            # Compute log prob and value
            log_prob, _ = self.policy.compute_log_probs(
                instruction, obs, action
            )
            value = self.value_model(instruction, obs)
            
            # Execute action
            next_obs, reward, done, step_info = self.env.step(action)
            
            # Store transition
            transition = Transition(
                instruction=instruction,
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                log_prob=log_prob.item(),
                value=value.item()
            )
            transitions.append(transition)
            
            total_reward += reward
            
            if verbose:
                print(f"\nStep {step + 1}:")
                print(f"  Action: {action}")
                print(f"  Reward: {reward:.3f}")
                print(f"  Value: {value.item():.3f}")
            
            obs = next_obs
            step += 1
        
        success = total_reward > 0.5  # Success if reward > 0.5
        
        if verbose:
            print(f"\n=== Episode End ===")
            print(f"Total Reward: {total_reward:.3f}")
            print(f"Success: {success}")
            print(f"Steps: {len(transitions)}")
        
        return Trajectory(
            transitions=transitions,
            total_reward=total_reward,
            success=success,
            length=len(transitions)
        )
    
    def collect_batch(
        self,
        num_episodes: int,
        temperature: float = 1.0,
        verbose: bool = False
    ) -> List[Trajectory]:
        """
        Collect batch of trajectories.
        
        Args:
            num_episodes: Number of episodes to collect
            temperature: Sampling temperature
            verbose: Print progress
            
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for i in range(num_episodes):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Episode {i+1}/{num_episodes}")
                print(f"{'='*60}")
            
            traj = self.collect_trajectory(
                temperature=temperature,
                verbose=verbose
            )
            trajectories.append(traj)
        
        return trajectories
    
    def compute_returns(self, trajectory: Trajectory) -> List[float]:
        """
        Compute discounted returns for trajectory.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            List of returns for each step
        """
        returns = []
        G = 0.0
        
        # Compute returns backwards
        for transition in reversed(trajectory.transitions):
            G = transition.reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def compute_advantages(
        self,
        trajectory: Trajectory,
        returns: List[float]
    ) -> List[float]:
        """
        Compute advantages using GAE.
        
        Args:
            trajectory: Episode trajectory
            returns: Computed returns
            
        Returns:
            List of advantages
        """
        advantages = []
        
        for i, (transition, G) in enumerate(zip(trajectory.transitions, returns)):
            advantage = G - transition.value
            advantages.append(advantage)
        
        return advantages
    
    def get_statistics(self, trajectories: List[Trajectory]) -> Dict:
        """
        Compute statistics for batch of trajectories.
        
        Args:
            trajectories: List of trajectories
            
        Returns:
            Dictionary of statistics
        """
        total_rewards = [t.total_reward for t in trajectories]
        lengths = [t.length for t in trajectories]
        successes = [t.success for t in trajectories]
        
        stats = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes),
            "num_episodes": len(trajectories)
        }
        
        return stats