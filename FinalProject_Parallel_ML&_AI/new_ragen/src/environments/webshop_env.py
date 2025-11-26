"""
WebShop Environment Wrapper
Clean implementation for RAGEN training
"""
import sys
from pathlib import Path

# Add WebShop to path (it's in the parent directory of week_06)
webshop_path = str(Path(__file__).parent.parent.parent.parent.parent / "WebShop")
if webshop_path not in sys.path:
    sys.path.insert(0, webshop_path)

import re
from typing import Dict, List, Tuple, Optional
from web_agent_site.envs import WebAgentTextEnv

class WebShopEnvironment:
    """
    Wrapper for WebShop environment with proper action parsing and reward shaping.
    """
    
    def __init__(self, num_products: int = 100):
        """
        Args:
            num_products: Number of products in WebShop (100, 1000, or full)
        """
        self.env = WebAgentTextEnv(
            observation_mode="text",
            num_products=num_products
        )
        self.num_products = num_products
        
        # Valid action types
        self.valid_actions = {
            "search": r"search\[(.*?)\]",
            "click": r"click\[(.*?)\]",
            "buy": r"buy now"
        }
        
    def reset(self) -> Tuple[str, Dict]:
        """Reset environment and return initial observation."""
        obs = self.env.reset()
        instruction = self._extract_instruction(obs)

        info = {
            "instruction": instruction,
            "step": 0,
            "done": False
        }

        return obs, info

    def reset_with_instruction(self, instruction: str) -> Tuple[str, Dict]:
        """
        Reset environment to a specific instruction.
        Note: WebShop doesn't support this directly, so we reset multiple times
        until we get the desired instruction or timeout.

        Args:
            instruction: Desired instruction to reset to

        Returns:
            obs: Initial observation
            info: Info dict with instruction
        """
        max_attempts = 100
        for _ in range(max_attempts):
            obs = self.env.reset()
            current_instruction = self._extract_instruction(obs)

            if current_instruction == instruction:
                info = {
                    "instruction": instruction,
                    "step": 0,
                    "done": False
                }
                return obs, info

        # If we can't find the exact instruction, just return a random one
        # and log a warning
        print(f"[WARNING] Could not reset to instruction: {instruction[:50]}... after {max_attempts} attempts")
        obs = self.env.reset()
        info = {
            "instruction": self._extract_instruction(obs),
            "step": 0,
            "done": False
        }
        return obs, info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action in environment."""
        parsed_action = self._parse_action(action)
        
        # WebShop returns: obs, reward, done, info (but info might be None)
        result = self.env.step(parsed_action)
        
        # Handle different return formats
        if len(result) == 4:
            obs, reward, done, env_info = result
        else:
            obs, reward, done = result
            env_info = {}
        
        # Ensure info is a dict
        if env_info is None:
            env_info = {}
        
        shaped_reward = self._shape_reward(reward, parsed_action, obs, done)
        
        # Create our info dict
        info = {
            "raw_reward": reward,
            "shaped_reward": shaped_reward,
            "action_type": self._get_action_type(parsed_action),
            "parsed_action": parsed_action
        }
        
        # Merge with env info if it exists
        if env_info:
            info.update(env_info)
        
        return obs, shaped_reward, done, info
    
    def _parse_action(self, action: str) -> str:
        """Parse and clean action string."""
        action = action.strip().lower()
        
        # Check each valid action pattern
        for action_type, pattern in self.valid_actions.items():
            if re.match(pattern, action):
                return action
        
        # Default to search if no valid pattern found
        return f"search[{action}]"
    
    def _get_action_type(self, action: str) -> str:
        """Determine type of action."""
        if "search[" in action:
            return "search"
        elif "click[" in action:
            return "click"
        elif "buy now" in action:
            return "buy"
        else:
            return "unknown"
    
    def _extract_instruction(self, obs: str) -> str:
        """Extract the user instruction from observation."""
        if "Instruction:" in obs:
            lines = obs.split("\n")
            for line in lines:
                if "Instruction:" in line:
                    return line.replace("Instruction:", "").strip()
        return ""
    
    def _shape_reward(self, reward: float, action: str, obs: str, done: bool) -> float:
        """Shape rewards to provide denser feedback."""
        shaped = reward
        action_type = self._get_action_type(action)
        
        if action_type == "search":
            search_term = self._extract_search_term(action)
            if len(search_term) > 2 and not search_term.startswith("search["):
                shaped += 0.1
        elif action_type == "click":
            shaped += 0.2
        elif action_type == "buy":
            shaped += 0.05
        elif action_type == "unknown":
            shaped -= 0.05
        
        return max(-0.1, min(2.0, shaped))
    
    def _extract_search_term(self, action: str) -> str:
        """Extract search term from search action."""
        match = re.search(r"search\[(.*?)\]", action)
        if match:
            return match.group(1)
        return ""
