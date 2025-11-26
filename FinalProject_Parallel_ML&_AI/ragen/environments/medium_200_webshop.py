"""
Medium200WebShop Environment - Stage 2 of Curriculum Learning

200 products:
- 100 complex synthetic products
- 100 real WebShop products (50 random + 50 diverse)

More complex than Stage 1:
- Longer product names with more attributes
- More keywords per product
- Mix of similar products (harder discrimination)
"""

import gym
from gym import spaces
import numpy as np
import json
from pathlib import Path


def load_products():
    """Load all 200 products from JSON files."""

    # Get the directory containing this file
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / "data"

    # Load real WebShop products (IDs 1-100)
    real_products_path = data_dir / "medium_webshop_real_products.json"
    with open(real_products_path) as f:
        real_products = json.load(f)

    # Load complex synthetic products (IDs 101-200)
    synthetic_products_path = data_dir / "medium_webshop_synthetic_products.json"
    with open(synthetic_products_path) as f:
        synthetic_products = json.load(f)

    # Combine into dict
    all_products = {}
    for prod in real_products + synthetic_products:
        all_products[prod['id']] = prod

    return all_products


class Medium200WebShop(gym.Env):
    """
    Medium complexity WebShop with 200 products - Stage 2 of curriculum.

    Features:
    - 200 diverse products (real + complex synthetic)
    - More nuanced keyword matching
    - Harder product discrimination
    - Bridge between simple and full WebShop
    """

    def __init__(self):
        super().__init__()

        # Load products
        self.products = load_products()
        self.num_products = len(self.products)

        self.action_space = spaces.Discrete(500)  # Not really used
        self.observation_space = spaces.Discrete(2000)  # Not really used

        self.reset()

    def reset(self):
        """Start new episode with random target product."""
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
            # Show top 10 search results
            results_str = "\n".join([
                f"{i}. {self.products[i]['name']}"
                for i in self.search_results[:10]
            ])
            return f"Search results:\n{results_str}\n\nWhich product do you want to view?"
        elif self.clicked_product:
            product_name = self.products[self.clicked_product]['name']
            return f"You are viewing: {product_name}\nDo you want to buy it?"
        else:
            return "Episode ended"

    def step(self, action_text):
        """
        Take action (as text) and return next state.

        Actions:
        - "search [keywords]" or just keywords
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
        """Handle search action with keyword matching."""
        # Extract keywords from action
        keywords = []
        stop_words = ['search', 'for', 'find', 'the', 'a', 'an', 'want', 'to', 'buy', 'get']
        for word in action_text.split():
            if word not in stop_words and len(word) > 1:
                keywords.append(word)

        # Find matching products
        self.search_results = []

        # Score each product by keyword overlap
        scores = []
        for prod_id, prod_data in self.products.items():
            prod_keywords = set(prod_data['keywords'])
            prod_name_words = set(prod_data['name'].lower().split())

            # Count matches in keywords and name
            keyword_matches = sum(1 for kw in keywords if kw in prod_keywords)
            name_matches = sum(1 for kw in keywords if kw in prod_name_words)

            total_score = keyword_matches * 2 + name_matches  # Keywords weighted more

            if total_score > 0:
                scores.append((prod_id, total_score))

        # Sort by score and take top results
        scores.sort(key=lambda x: x[1], reverse=True)
        self.search_results = [prod_id for prod_id, _ in scores[:30]]

        self.searched = True

        # Reward based on whether target is in results and position
        if self.target_product_id in self.search_results:
            target_pos = self.search_results.index(self.target_product_id)

            # Better reward for higher position
            if target_pos == 0:
                reward = 0.7  # Excellent! Top result
            elif target_pos < 5:
                reward = 0.5  # Great! Top 5
            elif target_pos < 10:
                reward = 0.3  # Good! Top 10
            else:
                reward = 0.1  # Found, but low rank

            info = {'search': 'success', 'found_target': True, 'position': target_pos}
        else:
            reward = -0.3  # Bad search, target not found
            info = {'search': 'fail', 'found_target': False}

        return reward, False, info

    def _handle_click(self, action_text):
        """Handle click action."""
        # Try to extract product number
        clicked_id = None

        # Try to find number in text
        import re
        numbers = re.findall(r'\d+', action_text)
        if numbers:
            clicked_id = int(numbers[0])

        # Check if it's a valid product in search results
        if clicked_id and clicked_id in self.search_results:
            self.clicked_product = clicked_id

            # Reward based on whether it's the target
            if clicked_id == self.target_product_id:
                reward = 0.5  # Correct product!
                info = {'click': 'correct'}
            else:
                reward = -0.3  # Wrong product
                info = {'click': 'wrong'}

            return reward, False, info
        else:
            # Invalid click
            reward = -0.4
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
            reward = -0.4
            info = {'buy': 'fail'}
            return reward, True, info


def get_expert_demonstrations(num_demos=200):
    """
    Generate expert demonstrations for Stage 2.

    Returns perfect trajectories for BC training.
    Use all 200 products.
    """
    products = load_products()
    demos = []

    for prod_id, prod_data in products.items():
        # Use first 3 keywords for search
        search_keywords = ' '.join(prod_data['keywords'][:3])

        demo = {
            'instruction': f"Find and buy {prod_data['name']}",
            'target_product': prod_id,
            'actions': [
                f"search {search_keywords}",
                f"click {prod_id}",
                "buy now"
            ],
            'expected_rewards': [0.7, 0.5, 1.0],  # Expected rewards
            'total_reward': 2.2,
            'success': True
        }
        demos.append(demo)

    return demos[:num_demos]


if __name__ == '__main__':
    # Test the environment
    print("="*60)
    print("Testing Medium200WebShop")
    print("="*60)

    env = Medium200WebShop()
    print(f"\nTotal products: {env.num_products}")

    # Test one episode
    print("\n" + "-"*60)
    print("Test Episode")
    print("-"*60)

    obs = env.reset()
    print(f"\nTarget: {obs['instruction'][:80]}...")
    print(f"Observation: {obs['observation'][:100]}...")

    # Search
    keywords = env.target_product['keywords'][:2]
    action = f"search {' '.join(keywords)}"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f}, Info: {info}")
    print(f"Observation: {obs[:150]}...")

    # Click
    action = f"click {env.target_product_id}"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f}, Info: {info}")
    print(f"Observation: {obs}")

    # Buy
    action = "buy now"
    print(f"\nAction: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.2f}, Done: {done}, Info: {info}")

    # Test expert demos
    print("\n" + "="*60)
    print("Expert Demonstrations Sample")
    print("="*60)

    demos = get_expert_demonstrations(num_demos=5)
    for i, demo in enumerate(demos):
        print(f"\nDemo {i+1}:")
        print(f"  Instruction: {demo['instruction'][:60]}...")
        print(f"  Actions: {demo['actions']}")
        print(f"  Total reward: {demo['total_reward']}")
