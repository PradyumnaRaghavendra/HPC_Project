"""
Expert demonstrations for WebShop environment
NOW USING REAL ASINS from actual WebShop database
"""

def get_webshop_expert_demos():
    """
    Returns REAL expert demonstration trajectories for WebShop.

    These are REAL ASINS from the WebShop database, correctly paired
    with their instructions.

    Each demo shows a complete successful shopping episode:
    1. Search for product
    2. Click correct product (REAL ASIN)
    3. Purchase

    Format matches what PPO trainer expects for warm-start.
    """
    # Load real expert demos
    from ragen.real_expert_demos import get_real_expert_demos
    real_demos = get_real_expert_demos()

    # Convert to format expected by BC warm-start
    demos = []
    for demo in real_demos:
        demos.append({
            'instruction': demo['instruction'],
            'actions': demo['actions']
        })

    return demos


def format_demos_for_ppo_warmstart():
    """
    Format expert demos for PPO warm-start training.

    Uses REAL ASINS from WebShop database.

    Returns list of (prompt, target_action) pairs for supervised learning.
    """
    demos = get_webshop_expert_demos()
    training_pairs = []

    for demo in demos:
        instruction = demo['instruction']

        # For each action in the trajectory, create a training example
        for i, action in enumerate(demo['actions']):
            if i == 0:
                # First action: just the instruction
                prompt = f"Task: {instruction}\n\nWhat should you do?"
            else:
                # Subsequent actions: show progress
                prev_actions = ', '.join(demo['actions'][:i])
                prompt = f"Task: {instruction}\n\nActions taken: {prev_actions}\n\nWhat should you do next?"

            training_pairs.append({
                'prompt': prompt,
                'target_action': action
            })

    return training_pairs


def print_demo_stats():
    """Print statistics about expert demonstrations"""
    demos = get_webshop_expert_demos()
    print(f"Total demos: {len(demos)}")
    print(f"Total training pairs: {len(format_demos_for_ppo_warmstart())}")

    # Show sample
    if demos:
        print("\nSample demo:")
        demo = demos[0]
        print(f"  Instruction: {demo['instruction']}")
        print(f"  Actions: {demo['actions']}")


if __name__ == '__main__':
    print_demo_stats()
