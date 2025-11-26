"""
WebShop Environment Wrapper for A*-PO Training
"""
import sys
import os
from pathlib import Path
import re
from typing import Dict, Tuple


class WebShopEnvironment:
    """
    Clean wrapper for WebShop environment.
    Handles action parsing and reward computation.
    """

    def __init__(self, num_products: int = 100):
        """
        Initialize WebShop environment.

        Args:
            num_products: Number of products (100 or 1000)
        """
        self.num_products = num_products

        # Import WebShop
        webshop_path = os.environ.get('WEBSHOP_PATH', '/root/WebShop')
        if webshop_path not in sys.path:
            sys.path.insert(0, webshop_path)

        try:
            from web_agent_site.envs import WebAgentTextEnv
            self.WebAgentTextEnv = WebAgentTextEnv
        except ImportError as e:
            print(f"❌ Failed to import WebShop: {e}")
            print(f"   Make sure WebShop is installed at {webshop_path}")
            raise

        # Create environment
        self.env = self.WebAgentTextEnv(
            observation_mode="text",
            num_products=num_products
        )

        print(f"✓ WebShop environment initialized ({num_products} products)")

    def reset(self) -> Tuple[str, Dict]:
        """
        Reset environment and return initial observation.

        Returns:
            observation: Text observation
            info: Dict with instruction and metadata
        """
        obs = self.env.reset()
        instruction = self._extract_instruction(obs)

        info = {
            'instruction': instruction,
            'step': 0,
            'done': False
        }

        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action string (e.g., "search[blue headphones]")

        Returns:
            observation: Next observation
            reward: Step reward
            done: Whether episode is done
            info: Additional information
        """
        # Parse and clean action
        parsed_action = self._parse_action(action)

        # Execute in WebShop
        result = self.env.step(parsed_action)

        if len(result) == 4:
            obs, reward, done, env_info = result
        else:
            obs, reward, done = result
            env_info = {}

        if env_info is None:
            env_info = {}

        # Create info dict
        info = {
            'raw_reward': float(reward),
            'action': action,
            'parsed_action': parsed_action
        }

        if env_info:
            info.update(env_info)

        return obs, float(reward), bool(done), info

    def _parse_action(self, action: str) -> str:
        """
        Parse and validate action string.

        Valid formats:
        - search[query]
        - click[button_name]
        - buy now

        If action doesn't match, tries to salvage it.
        """
        action = action.strip().lower()

        # Valid patterns
        search_pattern = r'search\[(.*?)\]'
        click_pattern = r'click\[(.*?)\]'
        buy_pattern = r'buy(\s+now)?'

        if re.match(search_pattern, action):
            return action
        elif re.match(click_pattern, action):
            return action
        elif re.match(buy_pattern, action):
            return 'buy now'
        else:
            # Try to salvage: wrap in search[...]
            if len(action) > 0:
                return f'search[{action}]'
            else:
                return 'search[product]'  # Default action

    def _extract_instruction(self, obs: str) -> str:
        """Extract instruction from observation."""
        if 'Instruction:' in obs:
            lines = obs.split('\n')
            for line in lines:
                if 'Instruction:' in line:
                    return line.replace('Instruction:', '').strip()
        return ""

    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
