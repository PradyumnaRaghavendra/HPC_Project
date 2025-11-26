"""
Uncertainty-Based Filtering (StarPO-S from RAGEN Paper)
Keep only high-variance prompts to prevent collapse
"""
import numpy as np
from typing import List, Dict


class UncertaintyFilter:
    """
    Filter rollouts based on reward variance.
    
    From RAGEN: "Training on high-variance prompts delays or 
    eliminates collapse by focusing on informative examples."
    """
    
    def __init__(self, keep_percent: float = 0.50):
        """
        Args:
            keep_percent: Fraction of prompts to keep (0.50 = top 50%)
        """
        self.keep_percent = keep_percent
        self.stats = {
            'total_batches': 0,
            'total_filtered': 0,
        }
    
    def should_filter(self, step: int) -> bool:
        """
        Decide if filtering should be enabled at this step.
        
        Strategy: Start filtering after 20 steps (warm-up period)
        """
        return step >= 20
    
    def filter_batch(self, batch: List[Dict], 
                    trajectories: List[Dict]) -> tuple:
        """
        Filter batch to keep only high-variance prompts.
        
        Args:
            batch: Original batch of tasks
            trajectories: Collected trajectories (parallel to batch)
        
        Returns:
            (filtered_batch, filtered_trajectories, stats)
        """
        if len(batch) <= 2:
            # Don't filter tiny batches
            return batch, trajectories, {'kept': len(batch), 'filtered': 0}
        
        # Group trajectories by prompt
        prompt_groups = {}
        for i, (task, traj) in enumerate(zip(batch, trajectories)):
            prompt = task.get('instruction', task.get('prompt', ''))
            if prompt not in prompt_groups:
                prompt_groups[prompt] = []
            prompt_groups[prompt].append((i, traj))
        
        # Calculate uncertainty (reward std) for each prompt
        uncertainties = []
        for prompt, group in prompt_groups.items():
            rewards = [traj['total_reward'] for _, traj in group]
            uncertainty = np.std(rewards) if len(rewards) > 1 else 0.0
            uncertainties.append((prompt, uncertainty, group))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top % most uncertain
        n_keep = max(1, int(len(uncertainties) * self.keep_percent))
        kept_groups = uncertainties[:n_keep]
        
        # Rebuild batch and trajectories
        kept_indices = []
        for _, _, group in kept_groups:
            for idx, _ in group:
                kept_indices.append(idx)
        
        filtered_batch = [batch[i] for i in kept_indices]
        filtered_trajectories = [trajectories[i] for i in kept_indices]
        
        # Update stats
        self.stats['total_batches'] += 1
        self.stats['total_filtered'] += len(batch) - len(filtered_batch)
        
        stats = {
            'kept': len(filtered_batch),
            'filtered': len(batch) - len(filtered_batch),
            'keep_ratio': len(filtered_batch) / len(batch),
        }
        
        return filtered_batch, filtered_trajectories, stats