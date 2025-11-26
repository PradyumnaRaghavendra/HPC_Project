"""
Medium-complexity WebShop for debugging training
Simpler than real WebShop but realistic multi-turn structure
"""
import random
from typing import Dict, Tuple, List
from .base import MultiTurnEnvironment


class MediumWebShopEnvironment(MultiTurnEnvironment):
    """
    Simplified WebShop with:
    - 10 products (manageable)
    - Simple string matching (not BM25)
    - Multi-turn: search → click → buy
    - Deterministic results
    - Clear partial rewards

    Target: 30-40% success rate to prove training works
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_turns = config.get('environment', {}).get('max_turns', 5)

        # Simple product database (10 products)
        self.products = [
            {'id': 0, 'name': 'wireless headphones', 'price': 29.99, 'keywords': ['wireless', 'headphones', 'audio', 'bluetooth']},
            {'id': 1, 'name': 'running shoes', 'price': 59.99, 'keywords': ['running', 'shoes', 'sport', 'athletic']},
            {'id': 2, 'name': 'coffee maker', 'price': 39.99, 'keywords': ['coffee', 'maker', 'kitchen', 'brew']},
            {'id': 3, 'name': 'laptop stand', 'price': 24.99, 'keywords': ['laptop', 'stand', 'desk', 'ergonomic']},
            {'id': 4, 'name': 'water bottle', 'price': 14.99, 'keywords': ['water', 'bottle', 'hydration', 'sport']},
            {'id': 5, 'name': 'yoga mat', 'price': 19.99, 'keywords': ['yoga', 'mat', 'exercise', 'fitness']},
            {'id': 6, 'name': 'phone charger', 'price': 12.99, 'keywords': ['phone', 'charger', 'usb', 'cable']},
            {'id': 7, 'name': 'desk lamp', 'price': 34.99, 'keywords': ['desk', 'lamp', 'light', 'led']},
            {'id': 8, 'name': 'backpack', 'price': 44.99, 'keywords': ['backpack', 'bag', 'travel', 'laptop']},
            {'id': 9, 'name': 'bluetooth speaker', 'price': 49.99, 'keywords': ['bluetooth', 'speaker', 'audio', 'wireless']},
        ]

        # Current state
        self.target_product = None
        self.search_results = []
        self.selected_product = None
        self.phase = 'search'  # search → click → buy
        self.action_history = []  # Track actions for repetition penalty

    def reset(self, task_data: Dict) -> str:
        """Start new task."""
        self.current_turn = 0
        self.history = []
        self.action_history = []  # Reset action history

        # Pick random product as target
        self.target_product = random.choice(self.products)
        self.search_results = []
        self.selected_product = None
        self.phase = 'search'

        # Task: find specific product
        instruction = f"I'm looking for a {self.target_product['name']}"

        return instruction

    def _search_products(self, query: str) -> List[Dict]:
        """Simple keyword-based search (deterministic)."""
        query_lower = query.lower()
        query_words = query_lower.split()

        # Score products by keyword matches
        scored = []
        for product in self.products:
            score = 0
            # Check if any query word matches product keywords
            for word in query_words:
                if word in product['keywords'] or word in product['name']:
                    score += 1

            if score > 0:
                scored.append((score, product))

        # Sort by score and return top 3
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:3]]

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action and return result."""
        self.current_turn += 1

        reward = 0.0
        done = False
        success = False
        observation = ""

        # Validate action
        if not action or not isinstance(action, str):
            info = {'success': False, 'phase': self.phase}
            return "Invalid action", 0.0, True, info

        action_lower = action.lower().strip()

        # Check for repetition penalty
        repetition_penalty = 0.0
        if action_lower in self.action_history:
            # Penalize repeating the same action
            repetition_penalty = -0.05
        self.action_history.append(action_lower)

        # PHASE 1: SEARCH
        if self.phase == 'search':
            # Expected: search[query]
            if action_lower.startswith('search[') and action_lower.endswith(']'):
                query = action_lower[7:-1].strip()

                # Search products
                self.search_results = self._search_products(query)

                if self.search_results:
                    # Show results
                    observation = "Search results:\n"
                    for i, prod in enumerate(self.search_results):
                        observation += f"[{i}] {prod['name']} - ${prod['price']}\n"

                    # Check if target is in results
                    if self.target_product in self.search_results:
                        reward = 0.4  # Good! Found target
                        self.phase = 'click'
                    else:
                        reward = 0.01  # Wrong search (reduced from 0.1 to prevent reward hacking)
                        observation += "\n(Target not found, try different search)"
                else:
                    reward = 0.01  # Correct format but no results (reduced from 0.05)
                    observation = "No results found. Try different keywords."
            else:
                reward = -0.1  # Wrong format
                observation = "Invalid search. Use: search[keywords]"

        # PHASE 2: CLICK
        elif self.phase == 'click':
            # Expected: click[idx]
            if action_lower.startswith('click[') and action_lower.endswith(']'):
                try:
                    idx_str = action_lower[6:-1].strip()
                    idx = int(idx_str)

                    if 0 <= idx < len(self.search_results):
                        self.selected_product = self.search_results[idx]

                        # Show product details
                        observation = f"Product: {self.selected_product['name']}\n"
                        observation += f"Price: ${self.selected_product['price']}\n"
                        observation += f"Description: {', '.join(self.selected_product['keywords'])}\n"

                        if self.selected_product == self.target_product:
                            reward = 0.4  # Correct product!
                            self.phase = 'buy'
                        else:
                            reward = 0.01  # Wrong product (reduced from 0.1 to prevent reward hacking)
                            observation += "\n(This is not the product you're looking for)"
                    else:
                        reward = -0.1  # Invalid index
                        observation = f"Invalid index. Choose 0-{len(self.search_results)-1}"
                except ValueError:
                    reward = -0.1
                    observation = "Invalid click format. Use: click[0], click[1], etc."
            else:
                reward = -0.1
                observation = "Invalid click. Use: click[index]"

        # PHASE 3: BUY
        elif self.phase == 'buy':
            # Expected: buy now
            if 'buy' in action_lower and 'now' in action_lower:
                if self.selected_product == self.target_product:
                    reward = 0.2  # Success bonus!
                    success = True
                    observation = "✓ Purchase successful!"
                else:
                    reward = -0.2  # Bought wrong product
                    observation = "✗ Wrong product purchased"
                done = True
            else:
                reward = -0.1
                observation = "Invalid buy command. Use: buy now"

        # Check max turns
        if self.current_turn >= self.max_turns:
            done = True
            if not success:
                observation += "\n(Max turns reached)"

        # Apply repetition penalty
        reward += repetition_penalty

        info = {
            'success': success,
            'phase': self.phase,
            'target': self.target_product['name'] if self.target_product else None,
            'action': action,
            'repetition_penalty': repetition_penalty
        }

        return observation, reward, done, info

    def compute_reward(self, trajectory: list) -> float:
        """Sum rewards from trajectory."""
        if not trajectory:
            return 0.0
        return sum(step.get('reward', 0.0) for step in trajectory)

    def render_text(self, state: str) -> str:
        """Format observation for model."""
        prompt = f"""Task: {state}

You are shopping online. Use these commands:
- search[keywords] - Search for products
- click[index] - Select a product from results (0, 1, 2)
- buy now - Purchase the selected product

>"""
        return prompt
