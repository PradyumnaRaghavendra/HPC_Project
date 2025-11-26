"""
Optimal Value Cache for A*-PO
Stores V*(x) - the best known reward for each prompt
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional

class OptimalValueCache:
    """
    Cache for storing optimal values V*(x) for each instruction.
    """
    
    def __init__(self, cache_file: str = "v_star_cache.json"):
        """
        Args:
            cache_file: Path to save/load V* values
        """
        self.cache_file = Path(cache_file)
        self.v_star: Dict[str, float] = {}
        
        # Load existing cache if available
        if self.cache_file.exists():
            self.load()
    
    def update(self, instruction: str, reward: float):
        """
        Update V*(x) if this reward is better than current best.
        
        Args:
            instruction: Task instruction
            reward: Achieved reward
        """
        current_best = self.v_star.get(instruction, -float('inf'))
        if reward > current_best:
            self.v_star[instruction] = reward
    
    def get(self, instruction: str, default: float = 0.0) -> float:
        """
        Get V*(x) for instruction.
        
        Args:
            instruction: Task instruction
            default: Default value if instruction not in cache
            
        Returns:
            Best known reward for this instruction
        """
        return self.v_star.get(instruction, default)
    
    def has(self, instruction: str) -> bool:
        """Check if instruction has V* value."""
        return instruction in self.v_star
    
    def save(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.v_star, f, indent=2)
    
    def load(self):
        """Load cache from disk."""
        with open(self.cache_file, 'r') as f:
            self.v_star = json.load(f)
    
    def get_statistics(self) -> Dict:
        """Get statistics about cached values."""
        if not self.v_star:
            return {
                "num_instructions": 0,
                "mean_v_star": 0.0,
                "max_v_star": 0.0,
                "coverage": 0.0
            }
        
        values = list(self.v_star.values())
        return {
            "num_instructions": len(self.v_star),
            "mean_v_star": np.mean(values),
            "max_v_star": np.max(values),
            "min_v_star": np.min(values),
            "std_v_star": np.std(values)
        }
