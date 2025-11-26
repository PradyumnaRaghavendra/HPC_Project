"""
Simple Mock WebShop Environment - Designed to Actually Work

Key simplifications:
1. Only 5 products total
2. Simple keyword matching (no complex search)
3. Products numbered 1-5 (no random ASINs)
4. Immediate rewards for correct actions
5. Natural language actions (no strict format)
"""

import gym
from gym import spaces
import numpy as np


class SimpleMockWebShop(gym.Env):
    """
    Ultra-simple WebShop mock for testing.

    Products:
    1. red shoes
    2. blue headphones
    3. black laptop
    4. white shirt
    5. green backpack
    """

    def __init__(self):
        super().__init__()

        # Define products (id, name, keywords)
        self.products = {
            1: {"name": "red shoes", "keywords": ["red", "shoes", "footwear"]},
            2: {"name": "blue headphones", "keywords": ["blue", "headphones", "audio"]},
            3: {"name": "black laptop", "keywords": ["black", "laptop", "computer"]},
            4: {"name": "white shirt", "keywords": ["white", "shirt", "clothing"]},
            5: {"name": "green backpack", "keywords": ["green", "backpack", "bag"]},
        }

        self.action_space = spaces.Discrete(10)  # Not really used
        self.observation_space = spaces.Discrete(100)  # Not really used

        self.reset()

    def reset(self, instruction=None):
        """Start new episode with random target product.

        Args:
            instruction: Optional instruction (ignored, for compatibility with MultiTurnPPOTrainer)
        """
        self.target_product_id = np.random.choice(list(self.products.keys()))
        self.target_product = self.products[self.target_product_id]

        self.current_step = 0
        self.max_steps = 3  # search, click, buy
        self.searched = False
        self.clicked_product = None
        self.search_results = []

        # Create instruction
        self.instruction = f"Find and buy {self.target_product['name']}"

        return {
            'observation': self.get_obs(),
            'instruction': self.instruction
        }

    def get_obs(self):
        """Get current observation."""
        if not self.searched:
            return f"Instruction: {self.instruction}\nWhat do you want to search for?"
        elif self.searched and not self.clicked_product:
            results_str = "\n".join([f"{i}. {self.products[i]['name']}" for i in self.search_results])
            return f"Search results:\n{results_str}\n\nWhich product do you want to view?"
        elif self.clicked_product:
            product_name = self.products[self.clicked_product]['name']
            return f"You are viewing: {product_name}\nDo you want to buy it?"
        else:
            return "Episode ended"

    def step(self, action_text):
        """
        Take action (as text) and return next state.

        Actions can be:
        - "search [keyword]" or just keywords
        - "click [number]" or just number
        - "buy" or "buy now"
        """
        action_text = action_text.lower().strip()
        reward = 0
        done = False
        info = {}

        self.current_step += 1

        # Phase 1: Search
        if not self.searched:
            reward, done, info = self._handle_search(action_text)

        # Phase 2: Click product
        elif self.searched and not self.clicked_product:
            reward, done, info = self._handle_click(action_text)

        # Phase 3: Buy
        elif self.clicked_product:
            reward, done, info = self._handle_buy(action_text)

        # Check max steps
        if self.current_step >= self.max_steps:
            done = True
            if reward <= 0:  # Didn't complete successfully
                reward = -1.0

        obs = self.get_obs()
        return obs, reward, done, info

    def _handle_search(self, action_text):
        """Handle search action."""
        # Extract keywords from action
        keywords = []
        for word in action_text.split():
            if word not in ['search', 'for', 'find', 'the', 'a', 'an']:
                keywords.append(word)

        # Find matching products
        self.search_results = []
        target_keywords = set(self.target_product['keywords'])

        for prod_id, prod_data in self.products.items():
            prod_keywords = set(prod_data['keywords'])
            # Check if any search keyword matches product keywords
            if any(kw in prod_keywords for kw in keywords):
                self.search_results.append(prod_id)

        self.searched = True

        # Reward based on whether target is in results
        if self.target_product_id in self.search_results:
            reward = 0.3  # Good search!
            info = {'search': 'success', 'found_target': True}
        else:
            reward = -0.1  # Bad search
            info = {'search': 'fail', 'found_target': False}

        return reward, False, info

    def _handle_click(self, action_text):
        """Handle click action."""
        # Try to extract product number
        clicked_id = None

        # Try direct number
        for char in action_text:
            if char.isdigit():
                clicked_id = int(char)
                break

        # Check if it's a valid product in search results
        if clicked_id and clicked_id in self.search_results:
            self.clicked_product = clicked_id

            # Reward based on whether it's the target
            if clicked_id == self.target_product_id:
                reward = 0.3  # Correct product!
                info = {'click': 'correct'}
            else:
                reward = -0.1  # Wrong product
                info = {'click': 'wrong'}

            return reward, False, info
        else:
            # Invalid click
            reward = -0.2
            info = {'click': 'invalid'}
            return reward, True, info  # End episode

    def _handle_buy(self, action_text):
        """Handle buy action."""
        # Check if action looks like "buy"
        if 'buy' in action_text:
            # Success if bought correct product
            if self.clicked_product == self.target_product_id:
                reward = 1.0  # SUCCESS!
                info = {'buy': 'success', 'correct_product': True}
            else:
                reward = -0.5  # Bought wrong product
                info = {'buy': 'success', 'correct_product': False}

            return reward, True, info
        else:
            # Didn't buy
            reward = -0.2
            info = {'buy': 'fail'}
            return reward, True, info


def get_expert_demonstrations():
    """
    Generate expert demonstrations for all products.

    Returns perfect trajectories for training.
    """
    products = {
        1: {"name": "red shoes", "search": "red shoes"},
        2: {"name": "blue headphones", "search": "blue headphones"},
        3: {"name": "black laptop", "search": "black laptop"},
        4: {"name": "white shirt", "search": "white shirt"},
        5: {"name": "green backpack", "search": "green backpack"},
    }

    demos = []

    for prod_id, prod_data in products.items():
        demo = {
            'instruction': f"Find and buy {prod_data['name']}",
            'target_product': prod_id,
            'actions': [
                f"search {prod_data['search']}",  # Search with keywords
                f"click {prod_id}",  # Click the product
                "buy now"  # Buy it
            ],
            'rewards': [0.3, 0.3, 1.0],  # Expected rewards
            'total_reward': 1.6,
            'success': True
        }
        demos.append(demo)

    return demos


if __name__ == '__main__':
    # Test the environment
    print("Testing SimpleMockWebShop\n")

    env = SimpleMockWebShop()

    # Test one episode
    obs = env.reset()
    print(f"Episode started!")
    print(f"Target: {obs['instruction']}")
    print(f"Observation: {obs['observation']}\n")

    # Search
    action = f"search {env.target_product['keywords'][0]}"
    print(f"Action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    print(f"Observation: {obs}\n")

    # Click
    action = f"click {env.target_product_id}"
    print(f"Action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    print(f"Observation: {obs}\n")

    # Buy
    action = "buy now"
    print(f"Action: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    print(f"\nFinal observation: {obs}")

    # Print expert demos
    print("\n" + "="*60)
    print("Expert Demonstrations:")
    demos = get_expert_demonstrations()
    for i, demo in enumerate(demos[:2]):  # Show first 2
        print(f"\nDemo {i+1}:")
        print(f"  Instruction: {demo['instruction']}")
        print(f"  Actions: {demo['actions']}")
        print(f"  Total reward: {demo['total_reward']}")
