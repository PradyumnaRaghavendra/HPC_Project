"""Expert demonstrations for Medium WebShop"""

def get_medium_expert_demos():
    """
    Expert demonstrations for medium WebShop.
    Simple 3-step trajectories: search → click → buy
    """
    demos = []

    # Products from medium_webshop.py
    products = [
        ('wireless headphones', 'wireless'),
        ('running shoes', 'running'),
        ('coffee maker', 'coffee'),
        ('laptop stand', 'laptop'),
        ('water bottle', 'water'),
        ('yoga mat', 'yoga'),
        ('phone charger', 'phone'),
        ('desk lamp', 'desk'),
        ('backpack', 'backpack'),
        ('bluetooth speaker', 'bluetooth'),
    ]

    for product_name, keyword in products:
        # Demo trajectory
        task_instruction = f"I'm looking for a {product_name}"

        # Turn 1: Search
        demos.append({
            'instruction': task_instruction,
            'action': f"search[{keyword}]",
            'turn': 0
        })

        # Turn 2: Click
        demos.append({
            'instruction': task_instruction,
            'action': "click[0]",
            'turn': 1
        })

        # Turn 3: Buy
        demos.append({
            'instruction': task_instruction,
            'action': "buy now",
            'turn': 2
        })

    return demos
