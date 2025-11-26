"""
Policy Model for RAGEN
Uses pretrained LM as base for action generation
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional

class RAGENPolicy(nn.Module):
    """
    Policy model that generates actions given state observations.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_length: int = 512,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: Hugging Face model identifier
            max_length: Maximum sequence length
            device: Device to run on
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.vocab_size = len(self.tokenizer)
        
    def format_prompt(
        self,
        instruction: str,
        observation: str,
        history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Format input as a prompt for the model.
        
        Args:
            instruction: Task instruction (what to buy)
            observation: Current environment state
            history: List of (action, observation) pairs
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a shopping agent. Your task is to find and purchase items.

Instruction: {instruction}

Current Page:
{observation}

You can take these actions:
- search[query] - Search for products
- click[item_id] - Click on an item to view details
- buy now - Purchase the current item

Generate ONLY the next action, nothing else.
Action:"""
        
        return prompt
    
    def generate_action(
        self,
        instruction: str,
        observation: str,
        history: List[Tuple[str, str]] = None,
        temperature: float = 1.0,
        sample: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate action for current state.
        
        Args:
            instruction: Task instruction
            observation: Current state
            history: Previous (action, obs) pairs
            temperature: Sampling temperature
            sample: Whether to sample or use greedy decoding
            
        Returns:
            action: Generated action string
            info: Additional info (logprobs, tokens, etc.)
        """
        prompt = self.format_prompt(instruction, observation, history)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():  # <-- Generation should always be no_grad
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temperature,
                do_sample=sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode action
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        action = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean action (remove extra text after action)
        action = self._clean_action(action)
        
        info = {
            "prompt_length": inputs.input_ids.shape[1],
            "generated_length": len(generated_ids),
            "raw_output": action
        }
        
        return action, info
    
    def _clean_action(self, action: str) -> str:
        """
        Clean generated action to extract only the action command.
        """
        action = action.strip()
        
        # Take only first line
        if "\n" in action:
            action = action.split("\n")[0]
        
        # Stop at common delimiters
        for delimiter in [".", ",", "Explanation:", "Reason:"]:
            if delimiter in action:
                action = action.split(delimiter)[0]
        
        return action.strip()
    
    def compute_log_probs(
        self,
        instruction: str,
        observation: str,
        action: str,
        history: List[Tuple[str, str]] = None,
        requires_grad: bool = False  # <-- FIX 1: Added requires_grad flag
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities for a given action.
        
        Args:
            instruction: Task instruction
            observation: Current state
            action: Action to compute logprobs for
            history: Previous interactions
            requires_grad: Set to True if we need to backward()
            
        Returns:
            log_probs: Log probability of action
            entropy: Entropy of action distribution
        """
        prompt = self.format_prompt(instruction, observation, history)
        full_text = prompt + " " + action
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # <-- FIX 1: Removed global no_grad, use set_grad_enabled -->
        # This lets the caller decide if gradients are needed.
        with torch.set_grad_enabled(requires_grad):
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get logprobs for action tokens only
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        # Ensure action_ids are within bounds
        if inputs.input_ids.shape[1] <= prompt_len:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        action_logits = logits[0, prompt_len-1:-1]
        action_ids = inputs.input_ids[0, prompt_len:]
        
        # Compute log probabilities
        log_probs_dist = torch.nn.functional.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs_dist.gather(1, action_ids.unsqueeze(1)).squeeze(1)
        
        # Sum log probs for full action
        total_log_prob = action_log_probs.sum()
        
        # Compute entropy
        probs_dist = torch.nn.functional.softmax(action_logits, dim=-1)
        entropy = -(probs_dist * log_probs_dist).sum(dim=-1).mean()
        
        return total_log_prob, entropy
    
    def save(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)