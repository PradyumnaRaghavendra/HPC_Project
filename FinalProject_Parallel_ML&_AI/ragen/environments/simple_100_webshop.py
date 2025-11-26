"""
Simple100WebShop Environment - Stage 1 of Curriculum Learning

100 synthetic but realistic products organized into clear categories.
Simple keyword matching, clear product differentiation.
"""

import gym
from gym import spaces
import numpy as np


# Generate 100 simple but realistic products
def generate_simple_products():
    """Generate 100 products across 5 categories, 20 products each."""

    products = {}
    product_id = 1

    # Category 1: Electronics (20 products)
    electronics = [
        ("red wireless mouse", ["red", "wireless", "mouse", "computer"]),
        ("blue mechanical keyboard", ["blue", "mechanical", "keyboard", "gaming"]),
        ("black laptop stand", ["black", "laptop", "stand", "desk"]),
        ("white bluetooth headphones", ["white", "bluetooth", "headphones", "audio"]),
        ("silver smartphone charger", ["silver", "smartphone", "charger", "cable"]),
        ("green led monitor", ["green", "led", "monitor", "screen"]),
        ("black gaming mouse pad", ["black", "gaming", "mouse", "pad"]),
        ("blue usb webcam", ["blue", "usb", "webcam", "camera"]),
        ("red portable speaker", ["red", "portable", "speaker", "bluetooth"]),
        ("black wireless earbuds", ["black", "wireless", "earbuds", "audio"]),
        ("white tablet case", ["white", "tablet", "case", "cover"]),
        ("silver hdmi cable", ["silver", "hdmi", "cable", "video"]),
        ("black phone holder", ["black", "phone", "holder", "stand"]),
        ("blue laptop bag", ["blue", "laptop", "bag", "carrying"]),
        ("red power bank", ["red", "power", "bank", "battery"]),
        ("green screen protector", ["green", "screen", "protector", "phone"]),
        ("white keyboard cover", ["white", "keyboard", "cover", "silicone"]),
        ("black mouse cable", ["black", "mouse", "cable", "cord"]),
        ("blue phone charger", ["blue", "phone", "charger", "fast"]),
        ("red gaming headset", ["red", "gaming", "headset", "microphone"]),
    ]

    for name, keywords in electronics:
        products[product_id] = {
            "name": name,
            "keywords": keywords,
            "category": "electronics",
            "price": np.random.randint(10, 100)
        }
        product_id += 1

    # Category 2: Clothing (20 products)
    clothing = [
        ("red cotton t-shirt", ["red", "cotton", "tshirt", "shirt"]),
        ("blue denim jeans", ["blue", "denim", "jeans", "pants"]),
        ("black leather jacket", ["black", "leather", "jacket", "coat"]),
        ("white running shoes", ["white", "running", "shoes", "sneakers"]),
        ("green hoodie sweatshirt", ["green", "hoodie", "sweatshirt", "pullover"]),
        ("gray wool socks", ["gray", "wool", "socks", "warm"]),
        ("brown leather belt", ["brown", "leather", "belt", "accessory"]),
        ("black baseball cap", ["black", "baseball", "cap", "hat"]),
        ("blue polo shirt", ["blue", "polo", "shirt", "casual"]),
        ("red winter gloves", ["red", "winter", "gloves", "warm"]),
        ("white dress shirt", ["white", "dress", "shirt", "formal"]),
        ("black cargo pants", ["black", "cargo", "pants", "pockets"]),
        ("green rain jacket", ["green", "rain", "jacket", "waterproof"]),
        ("yellow tank top", ["yellow", "tank", "top", "summer"]),
        ("gray sweatpants", ["gray", "sweatpants", "joggers", "comfortable"]),
        ("blue denim shorts", ["blue", "denim", "shorts", "summer"]),
        ("red flannel shirt", ["red", "flannel", "shirt", "plaid"]),
        ("black leather shoes", ["black", "leather", "shoes", "dress"]),
        ("white tennis shoes", ["white", "tennis", "shoes", "athletic"]),
        ("brown winter boots", ["brown", "winter", "boots", "snow"]),
    ]

    for name, keywords in clothing:
        products[product_id] = {
            "name": name,
            "keywords": keywords,
            "category": "clothing",
            "price": np.random.randint(15, 80)
        }
        product_id += 1

    # Category 3: Home & Kitchen (20 products)
    home_kitchen = [
        ("red coffee maker", ["red", "coffee", "maker", "machine"]),
        ("blue ceramic mug", ["blue", "ceramic", "mug", "cup"]),
        ("black frying pan", ["black", "frying", "pan", "cookware"]),
        ("white dish towel", ["white", "dish", "towel", "kitchen"]),
        ("green cutting board", ["green", "cutting", "board", "chopping"]),
        ("silver mixing bowl", ["silver", "mixing", "bowl", "stainless"]),
        ("red kitchen knife", ["red", "kitchen", "knife", "chef"]),
        ("blue water bottle", ["blue", "water", "bottle", "insulated"]),
        ("black toaster oven", ["black", "toaster", "oven", "cooking"]),
        ("white bed sheets", ["white", "bed", "sheets", "cotton"]),
        ("gray throw blanket", ["gray", "throw", "blanket", "soft"]),
        ("brown wooden spoon", ["brown", "wooden", "spoon", "cooking"]),
        ("red dish soap", ["red", "dish", "soap", "cleaning"]),
        ("blue storage box", ["blue", "storage", "box", "container"]),
        ("black trash can", ["black", "trash", "can", "garbage"]),
        ("white bath towel", ["white", "bath", "towel", "cotton"]),
        ("green plant pot", ["green", "plant", "pot", "planter"]),
        ("silver bottle opener", ["silver", "bottle", "opener", "tool"]),
        ("red oven mitt", ["red", "oven", "mitt", "kitchen"]),
        ("blue laundry basket", ["blue", "laundry", "basket", "hamper"]),
    ]

    for name, keywords in home_kitchen:
        products[product_id] = {
            "name": name,
            "keywords": keywords,
            "category": "home_kitchen",
            "price": np.random.randint(5, 60)
        }
        product_id += 1

    # Category 4: Sports & Outdoors (20 products)
    sports_outdoors = [
        ("red yoga mat", ["red", "yoga", "mat", "exercise"]),
        ("blue water bottle", ["blue", "water", "bottle", "sports"]),
        ("black resistance bands", ["black", "resistance", "bands", "fitness"]),
        ("white tennis ball", ["white", "tennis", "ball", "sport"]),
        ("green camping tent", ["green", "camping", "tent", "outdoor"]),
        ("orange foam roller", ["orange", "foam", "roller", "muscle"]),
        ("blue jump rope", ["blue", "jump", "rope", "cardio"]),
        ("red boxing gloves", ["red", "boxing", "gloves", "training"]),
        ("black gym bag", ["black", "gym", "bag", "sports"]),
        ("yellow frisbee disc", ["yellow", "frisbee", "disc", "outdoor"]),
        ("green hiking backpack", ["green", "hiking", "backpack", "trail"]),
        ("blue swimming goggles", ["blue", "swimming", "goggles", "pool"]),
        ("red bicycle helmet", ["red", "bicycle", "helmet", "safety"]),
        ("black dumbbell set", ["black", "dumbbell", "set", "weights"]),
        ("white golf balls", ["white", "golf", "balls", "sport"]),
        ("green sleeping bag", ["green", "sleeping", "bag", "camping"]),
        ("blue bike lock", ["blue", "bike", "lock", "security"]),
        ("red running armband", ["red", "running", "armband", "phone"]),
        ("black fishing rod", ["black", "fishing", "rod", "outdoor"]),
        ("orange life jacket", ["orange", "life", "jacket", "safety"]),
    ]

    for name, keywords in sports_outdoors:
        products[product_id] = {
            "name": name,
            "keywords": keywords,
            "category": "sports_outdoors",
            "price": np.random.randint(10, 90)
        }
        product_id += 1

    # Category 5: Books & Media (20 products)
    books_media = [
        ("red mystery novel", ["red", "mystery", "novel", "book"]),
        ("blue science fiction", ["blue", "science", "fiction", "book"]),
        ("black thriller book", ["black", "thriller", "book", "suspense"]),
        ("white cookbook recipes", ["white", "cookbook", "recipes", "cooking"]),
        ("green gardening guide", ["green", "gardening", "guide", "book"]),
        ("purple fantasy novel", ["purple", "fantasy", "novel", "magic"]),
        ("orange history book", ["orange", "history", "book", "historical"]),
        ("red romance novel", ["red", "romance", "novel", "love"]),
        ("blue travel guide", ["blue", "travel", "guide", "book"]),
        ("black biography book", ["black", "biography", "book", "life"]),
        ("white art book", ["white", "art", "book", "photography"]),
        ("green health guide", ["green", "health", "guide", "wellness"]),
        ("yellow children book", ["yellow", "children", "book", "kids"]),
        ("red comic book", ["red", "comic", "book", "graphic"]),
        ("blue poetry collection", ["blue", "poetry", "collection", "poems"]),
        ("black business book", ["black", "business", "book", "strategy"]),
        ("white self help", ["white", "self", "help", "motivation"]),
        ("green nature book", ["green", "nature", "book", "wildlife"]),
        ("red technology book", ["red", "technology", "book", "computers"]),
        ("blue music album", ["blue", "music", "album", "cd"]),
    ]

    for name, keywords in books_media:
        products[product_id] = {
            "name": name,
            "keywords": keywords,
            "category": "books_media",
            "price": np.random.randint(8, 40)
        }
        product_id += 1

    return products


class Simple100WebShop(gym.Env):
    """
    Simple WebShop with 100 products - Stage 1 of curriculum.

    Features:
    - 100 products across 5 categories
    - Simple keyword matching
    - Clear product names
    - Realistic but not complex
    """

    def __init__(self):
        super().__init__()

        # Generate products
        self.products = generate_simple_products()
        self.num_products = len(self.products)

        self.action_space = spaces.Discrete(200)  # Not really used
        self.observation_space = spaces.Discrete(1000)  # Not really used

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
        """Handle search action."""
        # Extract keywords from action
        keywords = []
        for word in action_text.split():
            if word not in ['search', 'for', 'find', 'the', 'a', 'an', 'want', 'to']:
                keywords.append(word)

        # Find matching products
        self.search_results = []
        target_keywords = set(self.target_product['keywords'])

        # Score each product by keyword overlap
        scores = []
        for prod_id, prod_data in self.products.items():
            prod_keywords = set(prod_data['keywords'])
            # Count matching keywords
            matches = sum(1 for kw in keywords if kw in prod_keywords)
            if matches > 0:
                scores.append((prod_id, matches))

        # Sort by match score and take top results
        scores.sort(key=lambda x: x[1], reverse=True)
        self.search_results = [prod_id for prod_id, _ in scores[:20]]

        self.searched = True

        # Reward based on whether target is in results
        if self.target_product_id in self.search_results:
            # Better reward if target is in top 5
            target_pos = self.search_results.index(self.target_product_id)
            if target_pos < 5:
                reward = 0.5  # Excellent search!
            else:
                reward = 0.3  # Good search
            info = {'search': 'success', 'found_target': True, 'position': target_pos}
        else:
            reward = -0.2  # Bad search
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
                # Handle multi-digit numbers
                try:
                    num_str = ''.join(c for c in action_text if c.isdigit())
                    if num_str:
                        clicked_id = int(num_str)
                except:
                    pass
                break

        # Check if it's a valid product in search results
        if clicked_id and clicked_id in self.search_results:
            self.clicked_product = clicked_id

            # Reward based on whether it's the target
            if clicked_id == self.target_product_id:
                reward = 0.5  # Correct product!
                info = {'click': 'correct'}
            else:
                reward = -0.2  # Wrong product
                info = {'click': 'wrong'}

            return reward, False, info
        else:
            # Invalid click
            reward = -0.3
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
            reward = -0.3
            info = {'buy': 'fail'}
            return reward, True, info


def get_expert_demonstrations(num_demos=100):
    """
    Generate expert demonstrations for Stage 1.

    Returns perfect trajectories for BC training.
    For simplicity, generate demos for first num_demos products.
    """
    products = generate_simple_products()
    demos = []

    # Generate demos for specified number of products
    product_ids = list(products.keys())[:num_demos]

    for prod_id in product_ids:
        prod_data = products[prod_id]

        # Use first 2 keywords for search
        search_keywords = ' '.join(prod_data['keywords'][:2])

        demo = {
            'instruction': f"Find and buy {prod_data['name']}",
            'target_product': prod_id,
            'actions': [
                f"search {search_keywords}",
                f"click {prod_id}",
                "buy now"
            ],
            'expected_rewards': [0.5, 0.5, 1.0],  # Expected rewards
            'total_reward': 2.0,
            'success': True
        }
        demos.append(demo)

    return demos


if __name__ == '__main__':
    # Test the environment
    print("="*60)
    print("Testing Simple100WebShop")
    print("="*60)

    env = Simple100WebShop()
    print(f"\nTotal products: {env.num_products}")

    # Test one episode
    print("\n" + "-"*60)
    print("Test Episode")
    print("-"*60)

    obs = env.reset()
    print(f"\nTarget: {obs['instruction']}")
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
        print(f"  Instruction: {demo['instruction']}")
        print(f"  Actions: {demo['actions']}")
        print(f"  Total reward: {demo['total_reward']}")
