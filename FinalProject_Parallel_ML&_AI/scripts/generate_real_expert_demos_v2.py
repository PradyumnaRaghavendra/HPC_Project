"""
Generate REAL expert demonstrations using actual WebShop ASIN-instruction pairs.
This uses the items_human_ins.json file which contains verified ASIN-instruction mappings.
"""
import json
from pathlib import Path


def generate_real_demos_from_data(num_demos=30):
    """
    Generate expert demonstrations using REAL WebShop ASIN-instruction pairs.

    Strategy:
    1. Load items_human_ins.json which has ASIN → instruction mappings
    2. For each ASIN + instruction pair, create demo:
       search[instruction] → click[ASIN] → buy now
    3. All ASINs are REAL and correctly paired with their instructions
    """

    # Load WebShop data
    webshop_data_path = Path(__file__).parent.parent.parent / "WebShop" / "data" / "items_human_ins.json"

    print("="*60)
    print("REAL EXPERT DEMONSTRATION GENERATOR v2")
    print("="*60)
    print(f"\nLoading WebShop data from: {webshop_data_path}")

    with open(webshop_data_path, 'r') as f:
        items_data = json.load(f)

    print(f"✓ Loaded {len(items_data)} ASINs from WebShop database")

    # Generate demonstrations
    expert_demos = []

    # Take first num_demos ASINs
    asins = list(items_data.keys())[:num_demos]

    print(f"\nGenerating {num_demos} expert demonstrations...")
    print("="*60)

    for i, asin in enumerate(asins, 1):
        # Get instruction for this ASIN
        asin_data = items_data[asin]

        # Handle list format (each ASIN can have multiple tasks)
        if isinstance(asin_data, list) and len(asin_data) > 0:
            task = asin_data[0]  # Use first task
            instruction = task.get('instruction', '')
        else:
            print(f"  ⚠️  Skipping {asin} - invalid data format")
            continue

        if not instruction:
            print(f"  ⚠️  Skipping {asin} - no instruction")
            continue

        # Create expert trajectory: search → click → buy
        demo = {
            'instruction': instruction,
            'asin': asin,
            'actions': [
                f'search[{instruction}]',
                f'click[{asin}]',
                'buy now'
            ],
            'rewards': [0.0, 0.0, 1.0],  # Standard reward structure
            'total_reward': 1.0,
            'success': True
        }

        expert_demos.append(demo)

        # Print progress
        instruction_preview = instruction[:60] + "..." if len(instruction) > 60 else instruction
        print(f"  {i:2d}. {asin} - {instruction_preview}")

    print("="*60)
    print(f"✓ Generated {len(expert_demos)} expert demonstrations")
    print("="*60)

    return expert_demos


def save_demos(demos, output_file='ragen/real_expert_demos.py'):
    """Save demonstrations as Python file"""

    print(f"\nSaving {len(demos)} demos to {output_file}...")

    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('REAL Expert Demonstrations for WebShop\n')
        f.write('Generated from actual WebShop ASIN-instruction pairs\n')
        f.write('All ASINs are REAL products from the WebShop database\n')
        f.write('All instructions are correctly paired with their ASINs\n')
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

    # Show a sample
    if demos:
        print(f"\nSample demonstration:")
        sample = demos[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  ASIN: {sample['asin']}")
        print(f"  Actions: {sample['actions']}")


if __name__ == '__main__':
    # Generate 30 real demonstrations
    demos = generate_real_demos_from_data(num_demos=30)

    # Save to file
    save_demos(demos)

    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Update ragen/multi_turn_ppo_trainer.py to use:")
    print("     from ragen.real_expert_demos import get_real_expert_demos")
    print("  2. Re-enable BC warm-start in ragen/train_ragen.py")
    print("  3. Deploy training v24 with real BC demonstrations")
    print("="*60)
