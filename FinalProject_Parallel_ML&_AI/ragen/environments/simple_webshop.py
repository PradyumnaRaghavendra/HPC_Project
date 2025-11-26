"""
Ultra-simplified WebShop for format learning
"""
import random
from typing import Dict, Tuple
from .base import MultiTurnEnvironment


class SimpleWebShopEnvironment(MultiTurnEnvironment):
    """
    Dead simple WebShop - just teach format.
    
    Task: "search for headphones"
    Correct: "search[headphones]" → reward 1.0
    Wrong format → reward 0.0
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_turns = 1
        self.target = None  # ← Initialize here!
    
    def reset(self, task_data: Dict) -> str:
        """Start new task."""
        self.current_turn = 0
        self.history = []
        
        # Simple search tasks
        queries = ['headphones', 'shoes', 'shirt', 'laptop', 'phone']
        self.target = random.choice(queries)  # ← Set target here!
        
        return f"search for {self.target}"
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Check if action matches format."""
        self.current_turn += 1
        
        # Initialize variables
        reward = 0.0
        success = False
        
        # Validate action exists
        if not action or not isinstance(action, str):
            info = {'success': False, 'target': self.target, 'action': str(action)}
            return "Invalid action", 0.0, True, info
        
        action_lower = action.lower().strip()
        
        # CASE 1: Perfect format - search[query]
        if action_lower.startswith('search[') and action_lower.endswith(']'):
            query = action_lower[7:-1].strip()
            
            if self.target in query:
                reward = 1.0  # Perfect!
                success = True
            else:
                reward = 0.3  # Right format, wrong query
                success = False
        
        # CASE 2: Almost right - just missing "search["
        elif action_lower.endswith(']'):
            # Model said "headphones]" instead of "search[headphones]"
            query = action_lower.rstrip(']').strip()
            
            if self.target in query:
                reward = 0.5  # Close! Just missing "search["
                success = False
            else:
                reward = 0.1  # Wrong query but right ending
                success = False
        
        # CASE 3: Completely wrong format
        else:
            reward = 0.0
            success = False
        
        info = {
            'success': success,
            'target': self.target,
            'action': action
        }
        
        return "Task complete", reward, True, info
    
    def compute_reward(self, trajectory: list) -> float:
        """Sum rewards from trajectory."""
        if not trajectory:
            return 0.0
        return sum(step.get('reward', 0.0) for step in trajectory)
    
    def render_text(self, state: str) -> str:
        """Absolute simplest prompt possible."""
        return f"""{state}

respond: search[word]

examples:
search[headphones]
search[shoes]
search[laptop]

>"""