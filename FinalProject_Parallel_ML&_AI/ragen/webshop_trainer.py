"""
Pure WebShop RL trainer - completely isolated from TinyZero
No conversational spillage, action-only training
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import re
import sys

# Import prompts
try:
    from ragen.prompts import get_webshop_system_prompt, get_webshop_user_prompt
except ImportError:
    # Fallback if running standalone
    def get_webshop_system_prompt():
        return """You are a WebShop agent. You MUST respond with ONLY ONE action.

VALID ACTIONS:
search[query] - Search for products
click[product_id] - Click a product (e.g., click[B07XYZ123])
buy now - Purchase current product
back - Go back

RULES:
1. Output ONLY the action, nothing else
2. No explanations, no reasoning, no text
3. One action per turn
4. If you output anything except a valid action, you FAIL

Example outputs:
search[laptop stand]
click[B07VNQN2V1]
buy now
back"""

    def get_webshop_user_prompt(task, observation):
        return f"""{task}

Current state: {observation}

Output your action now:"""


class WebShopAPOTrainer:
    """APO trainer specifically for WebShop - no TinyZero code"""
    
    def __init__(self, model, tokenizer, ref_model, env, config):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.env = env
        self.config = config
        
        # Action validation regex
        self.action_patterns = [
            r'^search\[.+\]$',
            r'^click\[B[0-9A-Z]+\]$',
            r'^buy now$',
            r'^back$'
        ]
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['apo']['learning_rate']
        )
        
    def is_valid_action(self, text: str) -> bool:
        """Check if output is a valid action format"""
        text = text.strip()
        return any(re.match(pattern, text) for pattern in self.action_patterns)
    
    def extract_action_only(self, raw_output: str) -> str:
        """Extract ONLY valid action, fail if not found"""
        # Remove everything before first valid action keyword
        raw_output = raw_output.strip()
        
        # Find first occurrence of action
        for keyword in ['search[', 'click[', 'buy now', 'back']:
            if keyword in raw_output.lower():
                idx = raw_output.lower().find(keyword)
                raw_output = raw_output[idx:]
                break
        
        # Try to extract search
        match = re.search(r'search\[([^\]]+)\]', raw_output, re.IGNORECASE)
        if match:
            query = match.group(1)
            # Basic cleaning only - no fallbacks
            query = query.replace('%20', ' ').replace('%2C', '').replace('%27', '')
            query = ' '.join(query.split()[:5])  # Max 5 words
            if query and len(query) > 2:
                return f"search[{query.strip()}]"
        
        # Try to extract click
        match = re.search(r'click\[(B[0-9A-Z]+)\]', raw_output, re.IGNORECASE)
        if match:
            return f"click[{match.group(1)}]"
        
        # Check for buy/back
        if 'buy' in raw_output.lower():
            return 'buy now'
        if 'back' in raw_output.lower():
            return 'back'
        
        # NO FALLBACK - return empty string to signal failure
        return ""
    
    def generate_with_format_enforcement(self, prompt: str) -> str:
        """Generate with strong bias toward action formats"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate short outputs only (actions are 5-20 tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,  # Very short - no room for conversation
                min_new_tokens=3,   # At least "search[x]"
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Discourage repeated tokens
                no_repeat_ngram_size=3   # Prevent loops
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def compute_format_penalty(self, action_str: str) -> float:
        """Heavy penalty for non-action outputs"""
        if not action_str or action_str == "":
            return -2.0  # Failed to extract action
        
        if not self.is_valid_action(action_str):
            return -1.0  # Invalid format
        
        # Check for generic/useless actions
        if action_str in ['search[products]', 'search[baby clothes]', 'search[smartphones]']:
            return -0.5  # Penalize generic queries
        
        return 0.0  # Valid action, no penalty
    
    def collect_trajectory(self, task: str) -> Dict:
        """Collect single trajectory with strict format enforcement"""
        trajectory = {
            'prompts': [],
            'responses': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'success': False
        }
        
        obs = self.env.reset(task)
        
        for turn in range(self.config['environment']['max_turns']):
            # Build prompt
            system_prompt = get_webshop_system_prompt()
            user_prompt = get_webshop_user_prompt(task, obs)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate
            raw_output = self.generate_with_format_enforcement(full_prompt)
            
            # Extract action
            action = self.extract_action_only(raw_output)
            
            # Compute format penalty FIRST
            format_penalty = self.compute_format_penalty(action)
            
            if action == "" or format_penalty < -1.5:
                # Complete failure - end episode
                trajectory['prompts'].append(full_prompt)
                trajectory['responses'].append(raw_output)
                trajectory['actions'].append('[INVALID]')
                trajectory['rewards'].append(-2.0)
                print(f"  ❌ Turn {turn}: Invalid action")
                print(f"     Raw: {raw_output[:80]}")
                break
            
            # Execute in environment
            obs, env_reward, done, info = self.env.step(action)
            
            # Compute total reward
            total_reward = env_reward + format_penalty
            
            # Small bonus for valid format
            if format_penalty == 0.0:
                total_reward += 0.1
            
            # Store
            trajectory['prompts'].append(full_prompt)
            trajectory['responses'].append(raw_output)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(total_reward)
            
            print(f"  Turn {turn}: {action[:50]} → reward={total_reward:.2f}")
            
            if done:
                trajectory['success'] = info.get('success', False)
                break
        
        return trajectory
    
    def compute_apo_loss(self, trajectories: List[Dict]) -> torch.Tensor:
        """Compute APO loss with format awareness"""
        all_log_probs = []
        all_advantages = []
        
        for traj in trajectories:
            if len(traj['rewards']) == 0:
                continue
            
            # Compute advantages (simple: rewards - mean)
            rewards = torch.tensor(traj['rewards'])
            advantages = rewards - rewards.mean()
            
            # Get log probs for each response
            for prompt, response in zip(traj['prompts'], traj['responses']):
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
                response_ids = self.tokenizer.encode(response, return_tensors='pt')
                
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                response_ids = response_ids.to(self.model.device)
                
                # Forward pass
                with torch.set_grad_enabled(True):
                    outputs = self.model(**inputs, labels=response_ids)
                    log_prob = -outputs.loss  # Negative loss = log prob
                
                all_log_probs.append(log_prob)
            
            all_advantages.extend(advantages.tolist())
        
        if not all_log_probs:
            return torch.tensor(0.0, device=self.model.device)
        
        # Stack
        log_probs = torch.stack(all_log_probs)
        advantages = torch.tensor(all_advantages, device=self.model.device)
        
        # APO loss: -log_prob * advantage
        loss = -(log_probs * advantages).mean()
        
        return loss
    
    def train_step(self, batch_tasks: List[str]) -> Dict:
        """Single training step"""
        # Collect trajectories
        trajectories = []
        total_reward = 0
        num_success = 0
        
        print(f"\n🎯 Collecting {len(batch_tasks)} trajectories...")
        for task in batch_tasks:
            traj = self.collect_trajectory(task)
            trajectories.append(traj)
            total_reward += sum(traj['rewards'])
            if traj['success']:
                num_success += 1
        
        # Compute loss
        loss = self.compute_apo_loss(trajectories)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'reward': total_reward / len(batch_tasks),
            'success_rate': num_success / len(batch_tasks)
        }