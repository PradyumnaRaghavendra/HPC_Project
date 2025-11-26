"""
Generate REAL expert demonstrations using actual WebShop ASINs.
This script interacts with the WebShop environment to collect valid trajectories.
"""

import sys
import json
sys.path.insert(0, '/Users/nikhilpandey/Self_Improving_AI/neu-self-improve-ai/week_06')

from ragen.environments.webshop import WebShopEnvironment


def generate_real_demos(num_demos=30):
    """
    Generate expert demonstrations using REAL WebShop data.

    Strategy:
    1. Load actual WebShop tasks
    2. For each task, perform search with the instruction
    3. Extract REAL ASINs from search results
    4. Create trajectory: search[query] → click[REAL_ASIN] → buy now
    """

    # Initialize WebShop environment
    config = {
        'environment': {
            'type': 'webshop',
            'max_turns': 5,
            'num_products': 100
        },
        'data': {
            'train_size': 100,
            'eval_size': 25
        }
    }

    print("Initializing WebShop environment...")
    env = WebShopEnvironment(config)

    # Load actual tasks
    import json
    tasks_file = '/Users/nikhilpandey/Self_Improving_AI/neu-self-improve-ai/week_06/data/webshop/items_shuffle_1000.json'

    print(f"Loading tasks from {tasks_file}...")
    with open(tasks_file, 'r') as f:
        all_tasks = json.load(f)

    print(f"Loaded {len(all_tasks)} tasks")

    expert_demos = []
    successful_demos = 0

    for i, task in enumerate(all_tasks[:num_demos]):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{num_demos}: {task.get('instruction', task.get('query', 'N/A'))[:80]}")
        print(f"{'='*60}")

        # Reset environment with this task
        state = env.reset(task)

        # Extract search query from instruction
        instruction = task.get('instruction', task.get('query', ''))

        # Step 1: Search
        search_action = f"search[{instruction}]"
        print(f"  Action 1: {search_action}")

        next_state, reward_1, done_1, info_1 = env.step(search_action)
        print(f"    Reward: {reward_1}, Done: {done_1}")

        # Check if we got search results
        if hasattr(next_state, 'get') and 'results' in str(next_state):
            # Extract ASINs from state
            state_str = str(next_state)

            # Look for product ASINs (format: B followed by 9 alphanumeric characters)
            import re
            asin_pattern = r'B[A-Z0-9]{9}'
            found_asins = re.findall(asin_pattern, state_str)

            if found_asins:
                # Use the first REAL ASIN
                real_asin = found_asins[0]
                print(f"  Found REAL ASIN: {real_asin}")

                # Step 2: Click on real ASIN
                click_action = f"click[{real_asin}]"
                print(f"  Action 2: {click_action}")

                next_state_2, reward_2, done_2, info_2 = env.step(click_action)
                print(f"    Reward: {reward_2}, Done: {done_2}")

                # Step 3: Buy now
                buy_action = "buy now"
                print(f"  Action 3: {buy_action}")

                next_state_3, reward_3, done_3, info_3 = env.step(buy_action)
                print(f"    Reward: {reward_3}, Done: {done_3}")

                # Calculate total reward
                total_reward = reward_1 + reward_2 + reward_3
                success = total_reward > 0.5

                print(f"\n  Total Reward: {total_reward}")
                print(f"  Success: {'✓ YES' if success else '✗ NO'}")

                # Save demo
                demo = {
                    'instruction': instruction,
                    'asin': real_asin,
                    'actions': [
                        search_action,
                        click_action,
                        buy_action
                    ],
                    'rewards': [reward_1, reward_2, reward_3],
                    'total_reward': total_reward,
                    'success': success
                }

                expert_demos.append(demo)

                if success:
                    successful_demos += 1
                    print(f"  ✓ Added successful demo ({successful_demos} total)")
                else:
                    print(f"  + Added demo (partial success)")
            else:
                print(f"  ✗ No ASINs found in search results")
        else:
            print(f"  ✗ Search failed - no results")

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total demos collected: {len(expert_demos)}")
    print(f"Successful demos: {successful_demos}")
    print(f"Success rate: {successful_demos/len(expert_demos)*100:.1f}%")

    return expert_demos


def save_demos(demos, output_file='ragen/real_expert_demos.py'):
    """Save demonstrations as Python file"""

    print(f"\nSaving {len(demos)} demos to {output_file}...")

    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('REAL Expert Demonstrations for WebShop\n')
        f.write('Generated from actual WebShop environment interactions\n')
        f.write('All ASINs are REAL products from the WebShop database\n')
        f.write('"""\n\n')
        f.write('def get_real_expert_demos():\n')
        f.write('    """Expert trajectories with REAL WebShop ASINs"""\n')
        f.write('    return [\n')

        for demo in demos:
            f.write('        {\n')
            f.write(f"            'instruction': {repr(demo['instruction'])},\n")
            f.write(f"            'asin': {repr(demo['asin'])},\n")
            f.write(f"            'actions': {demo['actions']},\n")
            f.write(f"            'rewards': {demo['rewards']},\n")
            f.write(f"            'total_reward': {demo['total_reward']},\n")
            f.write(f"            'success': {demo['success']}\n")
            f.write('        },\n')

        f.write('    ]\n')

    print(f"✓ Saved to {output_file}")


if __name__ == '__main__':
    print("="*60)
    print("REAL EXPERT DEMONSTRATION GENERATOR")
    print("="*60)
    print()

    # Generate 30 real demonstrations
    demos = generate_real_demos(num_demos=30)

    # Save to file
    save_demos(demos)

    print("\n✓ Done! You can now use these real demos for BC training.")
    print("  Update ragen/multi_turn_ppo_trainer.py to use:")
    print("  from ragen.real_expert_demos import get_real_expert_demos")
