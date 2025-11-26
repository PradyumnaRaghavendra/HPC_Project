"""
Real WebShop environment wrapper for RAGEN
Uses the actual WebShop framework
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add WebShop to Python path
WEBSHOP_PATH = Path(__file__).parent.parent.parent.parent / "WebShop"
sys.path.insert(0, str(WEBSHOP_PATH))

from web_agent_site.envs import WebAgentTextEnv
from .base import MultiTurnEnvironment


class WebShopEnvironment(MultiTurnEnvironment):
    """
    Wrapper around the REAL WebShop environment.
    
    This uses the actual Princeton WebShop benchmark with:
    - Real product database (1000 products)
    - Real search functionality
    - Real attribute matching
    - Official WebShop tasks
    """
    
    def __init__(self, config: Dict):
        """Initialize real WebShop environment"""
        super().__init__(config)
        
        # WebShop configuration
        self.max_turns = config.get('environment', {}).get('max_turns', 10)
        num_products = config.get('environment', {}).get('num_products', 100)
        
        print(f"Initializing REAL WebShop Environment...")
        print(f"  Max turns: {self.max_turns}")
        print(f"  Product count: {num_products}")
        
        # Create real WebShop environment
        self.env = WebAgentTextEnv(
            observation_mode='text',  # Text-based observations
            num_products=num_products,
        )
        
        # Episode state
        self.current_instruction = None
        self.session = None
        
        print("✓ REAL WebShop Environment initialized!")
    
    def reset(self, task_data: Dict) -> str:
        """
        Start new WebShop task.
        
        Args:
            task_data: Can contain 'session' for specific task
        
        Returns:
            Initial observation from WebShop
        """
        self.current_turn = 0
        self.history = []

        # Handle both dict (full task data) and str (instruction only)
        if isinstance(task_data, str):
            # Just an instruction string - no session
            self.session = None
        else:
            # Full task data dict
            self.session = task_data.get('session', None)
        
        try:
            # Reset WebShop environment
            obs_tuple = self.env.reset(session=self.session)
            
            # WebShop returns (observation, info) tuple
            if isinstance(obs_tuple, tuple):
                obs = obs_tuple[0]
            else:
                obs = obs_tuple
            
            # Validate observation
            if obs is None or not isinstance(obs, str):
                print(f"⚠️  Invalid observation from reset: {obs}")
                obs = "Welcome to WebShop! Please search for products."
            
            # Extract instruction from observation
            if '[SEP]' in obs:
                parts = obs.split('[SEP]')
                if len(parts) >= 3:
                    self.current_instruction = parts[2].strip()
            
            return obs
            
        except Exception as e:
            print(f"⚠️  Error in reset: {e}")
            import traceback
            traceback.print_exc()
            return "Welcome to WebShop! Please search for products."
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action in real WebShop."""
        self.current_turn += 1
        
        # Initialize ALL variables FIRST - before any try block
        obs = "Error occurred"
        reward = 0.0
        done = True
        info = {
            'turn': self.current_turn,
            'max_turns': self.max_turns,
            'timeout': False,
            'success': False,
            'error': None
        }
        
        # Validate action
        if action is None or (isinstance(action, str) and not action.strip()):
            print(f"⚠️ Invalid action at turn {self.current_turn}: {repr(action)}")
            info['error'] = 'invalid_action'
            return "Invalid action. Please try again.", 0.0, True, info
        
        # Execute action in WebShop
        try:
            result = self.env.step(action)
            
            # Parse result
            if isinstance(result, tuple) and len(result) >= 3:
                obs = result[0] if result[0] is not None else "No observation"
                reward = float(result[1]) if result[1] is not None else 0.0
                done = bool(result[2]) if result[2] is not None else True
                
                if len(result) >= 4 and isinstance(result[3], dict):
                    info.update(result[3])
            else:
                print(f"⚠️ Unexpected result type: {type(result)}")
                info['error'] = 'unexpected_result'
                
        except Exception as e:
            print(f"⚠️ WebShop step error: {e}")
            info['error'] = str(e)
            # reward, done, obs already initialized above
        
        # Check timeout
        if self.current_turn >= self.max_turns and not done:
            done = True
            info['timeout'] = True
        
        # Check success
        if reward > 0:
            info['success'] = True
        
        # Store in history
        try:
            self.add_to_history(obs, action, reward)
        except Exception as e:
            print(f"⚠️ Error adding to history: {e}")
        
        return obs, reward, done, info
    
    def compute_reward(self, trajectory: list) -> float:
        """
        Calculate reward from trajectory.
        WebShop provides rewards automatically.
        """
        if not trajectory:
            return 0.0
        
        total = sum(step.get('reward', 0.0) for step in trajectory)
        return float(total)
    
    def render_text(self, state: str) -> str:
        """
        Clean observation text including product listings.
        System prompt in agent_trainer.py handles instructions.

        WebShop state format: "Instruction [SEP] Observation [SEP] Task [SEP] Products..."
        Products are formatted as: [SEP] Page info [SEP] Next > [SEP] B0XXX [SEP] Name [SEP] Price [SEP] ...
        """
        if state is None or not state:
            return "Welcome to WebShop! Search for products to begin."

        state_str = str(state)

        # Parse WebShop state format
        if '[SEP]' in state_str:
            parts = state_str.split('[SEP]')

            # Extract components
            instruction = parts[0].strip() if len(parts) > 0 else ""
            observation = parts[1].strip() if len(parts) > 1 else state_str
            task_detail = parts[2].strip() if len(parts) > 2 else ""

            # Build clean prompt
            prompt_parts = []

            if task_detail:
                prompt_parts.append(f"Task: {task_detail}")
            elif instruction:
                prompt_parts.append(f"Task: {instruction}")

            if observation:
                prompt_parts.append(f"\n{observation}")

            # CRITICAL FIX: Include product listings from parts[3+]
            # WebShop returns products as: Page info [SEP] Next > [SEP] ProductID [SEP] Name [SEP] Price [SEP] ...
            if len(parts) > 3:
                # Add all remaining parts (product listings, navigation, etc.)
                products_text = ' [SEP] '.join(part.strip() for part in parts[3:] if part.strip())
                if products_text:
                    prompt_parts.append(f"\n{products_text}")

            return '\n'.join(prompt_parts) if prompt_parts else observation

        # Fallback for non-standard format
        return state_str[:500]  # Truncate very long states
