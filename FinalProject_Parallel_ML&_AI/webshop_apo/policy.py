"""
Policy Model for WebShop A*-PO Training
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List


class PolicyModel:
    """
    Wrapper for LLM policy model.
    Handles generation and training for WebShop tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device

        print(f"Loading policy model: {model_name}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print(f"✓ Model loaded on {device}")

    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    @torch.no_grad()
    def generate_action(
        self,
        instruction: str,
        observation: str,
        previous_actions: Optional[List[str]] = None,
        temperature: float = 0.9,
        do_sample: bool = True,
        top_p: float = 0.95,
        max_new_tokens: int = 50
    ) -> str:
        """
        Generate next action given instruction, observation, and history.

        Returns a WebShop action like:
        - "search[blue headphones]"
        - "click[B09QKP7XQL]"
        - "buy now"
        """
        # Build prompt
        if previous_actions is None:
            previous_actions = []

        prompt = self._build_action_prompt(instruction, observation, previous_actions)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract action (first line)
        action = generated_text.strip().split('\n')[0].strip()

        # Clean up action
        action = self._clean_action(action)

        return action

    def _build_action_prompt(
        self,
        instruction: str,
        observation: str,
        previous_actions: List[str]
    ) -> str:
        """
        Build prompt for action generation.

        Format:
        You are shopping on WebShop. Your goal: {instruction}

        History:
        {previous actions}

        Current page:
        {observation}

        What action should you take? Respond with ONE action:
        - search[query] to search for products
        - click[button] to click a button/link
        - buy now to purchase the current product

        Action:
        """
        prompt = f"""You are shopping on WebShop. Your goal: {instruction}

"""

        if previous_actions:
            prompt += "Previous actions:\n"
            for i, action in enumerate(previous_actions[-5:], 1):  # Last 5 actions
                prompt += f"{i}. {action}\n"
            prompt += "\n"

        # Truncate observation to fit in context
        obs_truncated = observation[:500] if len(observation) > 500 else observation

        prompt += f"""Current page:
{obs_truncated}

What action should you take? Choose ONE action:
- search[query] - search for products
- click[button] - click a button or link
- buy now - purchase current product

Action:"""

        return prompt

    def _clean_action(self, action: str) -> str:
        """
        Clean and validate generated action.

        Ensures action follows WebShop format:
        - search[...]
        - click[...]
        - buy now
        """
        action = action.strip().lower()

        # Remove common prefixes
        prefixes = ['action:', 'response:', 'output:']
        for prefix in prefixes:
            if action.startswith(prefix):
                action = action[len(prefix):].strip()

        # Extract first valid action if multiple lines
        if '\n' in action:
            action = action.split('\n')[0].strip()

        # Validate format
        if not any(action.startswith(x) for x in ['search[', 'click[', 'buy']):
            # Try to salvage: if it looks like a search query, wrap it
            if len(action) > 0 and not action.startswith('['):
                action = f'search[{action}]'

        return action

    def save(self, path: str):
        """Save model checkpoint."""
        print(f"Saving policy to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print("✓ Policy saved")

    def load(self, path: str):
        """Load model checkpoint."""
        print(f"Loading policy from {path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        print("✓ Policy loaded")


class ReferenceModel:
    """
    Reference model for V* computation and KL regularization.
    Frozen copy of the initial policy.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device

        print(f"Loading reference model: {model_name}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        print(f"✓ Reference model loaded and frozen on {device}")

    @torch.no_grad()
    def generate_action(self, *args, **kwargs) -> str:
        """Generate action (same interface as PolicyModel)."""
        # Use same generation logic as PolicyModel
        # For simplicity, wrap in a PolicyModel instance
        temp_policy = PolicyModel.__new__(PolicyModel)
        temp_policy.model = self.model
        temp_policy.tokenizer = self.tokenizer
        temp_policy.device = self.device
        temp_policy.model_name = self.model_name

        return temp_policy.generate_action(*args, **kwargs)
