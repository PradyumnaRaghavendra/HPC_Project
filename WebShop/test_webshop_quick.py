"""
Quick test if WebShop works despite pip errors
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Attempting to import WebShop...")
    from web_agent_site.envs import WebAgentTextEnv
    print("✓ WebShop imported successfully!")
    
    print("\nCreating environment...")
    env = WebAgentTextEnv(observation_mode='text', num_products=100)
    print("✓ Environment created!")
    
    print("\nResetting environment...")
    obs = env.reset()
    print("✓ Reset successful!")
    print(f"\nObservation preview (first 300 chars):\n{obs[:300]}...")
    
    print("\n" + "="*60)
    print("✅ WEBSHOP WORKS PERFECTLY!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nWebShop doesn't work - we'll use alternative approach")
