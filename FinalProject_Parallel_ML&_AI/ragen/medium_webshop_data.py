"""
Synthetic data generation for Medium WebShop
Creates simple training examples with perfect demonstrations
"""
import json
from typing import List, Dict


# Products from medium_webshop.py
PRODUCTS = [
    {'id': 0, 'name': 'wireless headphones', 'keywords': ['wireless', 'headphones', 'audio', 'bluetooth']},
    {'id': 1, 'name': 'running shoes', 'keywords': ['running', 'shoes', 'sport', 'athletic']},
    {'id': 2, 'name': 'coffee maker', 'keywords': ['coffee', 'maker', 'kitchen', 'brew']},
    {'id': 3, 'name': 'laptop stand', 'keywords': ['laptop', 'stand', 'desk', 'ergonomic']},
    {'id': 4, 'name': 'water bottle', 'keywords': ['water', 'bottle', 'hydration', 'sport']},
    {'id': 5, 'name': 'yoga mat', 'keywords': ['yoga', 'mat', 'exercise', 'fitness']},
    {'id': 6, 'name': 'phone charger', 'keywords': ['phone', 'charger', 'usb', 'cable']},
    {'id': 7, 'name': 'desk lamp', 'keywords': ['desk', 'lamp', 'light', 'led']},
    {'id': 8, 'name': 'backpack', 'keywords': ['backpack', 'bag', 'travel', 'laptop']},
    {'id': 9, 'name': 'bluetooth speaker', 'keywords': ['bluetooth', 'speaker', 'audio', 'wireless']},
]


def generate_training_data(num_samples: int = 100) -> List[Dict]:
    """Generate training tasks (just product names for now)."""
    import random

    tasks = []
    for i in range(num_samples):
        product = random.choice(PRODUCTS)
        tasks.append({
            'id': i,
            'instruction': f"I'm looking for a {product['name']}",
            'target_product': product['name'],
            'target_id': product['id']
        })

    return tasks


def generate_expert_demos() -> List[Dict]:
    """Generate expert demonstrations for each product."""
    demos = []

    for product in PRODUCTS:
        # Use main keyword for search
        main_keyword = product['keywords'][0]

        # Perfect trajectory: search → click → buy
        demo = {
            'task': f"I'm looking for a {product['name']}",
            'trajectory': [
                {
                    'turn': 0,
                    'action': f"search[{main_keyword}]",
                    'explanation': f"Search for {main_keyword} to find {product['name']}"
                },
                {
                    'turn': 1,
                    'action': "click[0]",
                    'explanation': f"Click on first result (should be {product['name']})"
                },
                {
                    'turn': 2,
                    'action': "buy now",
                    'explanation': "Purchase the product"
                }
            ],
            'target_product': product['name']
        }
        demos.append(demo)

    return demos


def create_sft_dataset() -> List[Dict]:
    """Create supervised fine-tuning dataset from expert demos."""
    demos = generate_expert_demos()
    sft_data = []

    for demo in demos:
        task = demo['task']

        # Create examples for each step
        for step in demo['trajectory']:
            example = {
                'prompt': f"{task}\n\nYou are shopping online. Use these commands:\n- search[keywords]\n- click[index]\n- buy now\n\n>",
                'completion': step['action'],
                'task': task,
                'turn': step['turn']
            }
            sft_data.append(example)

    return sft_data


if __name__ == '__main__':
    # Generate training data
    train_data = generate_training_data(100)
    eval_data = generate_training_data(25)

    print(f"Generated {len(train_data)} training samples")
    print(f"Generated {len(eval_data)} eval samples")

    # Save to files
    with open('data/medium_webshop_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open('data/medium_webshop_eval.json', 'w') as f:
        json.dump(eval_data, f, indent=2)

    # Generate expert demos
    demos = generate_expert_demos()
    print(f"Generated {len(demos)} expert demonstrations")

    with open('data/medium_webshop_demos.json', 'w') as f:
        json.dump(demos, f, indent=2)

    # Generate SFT dataset
    sft_data = create_sft_dataset()
    print(f"Generated {len(sft_data)} SFT examples")

    with open('data/medium_webshop_sft.json', 'w') as f:
        json.dump(sft_data, f, indent=2)

    print("\n✓ All data generated successfully!")
    print("  - data/medium_webshop_train.json")
    print("  - data/medium_webshop_eval.json")
    print("  - data/medium_webshop_demos.json")
    print("  - data/medium_webshop_sft.json")
