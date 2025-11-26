"""
Value Model for RAGEN
Estimates V(s) for states during training
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ValueModel(nn.Module):
    """
    Value model that estimates state values.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_length: int = 512,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: Base model for encoding
            max_length: Max sequence length
            device: Device to run on
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Load encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # Value head
        hidden_size = self.encoder.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        ).to(device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, instruction: str, observation: str) -> torch.Tensor:
        """
        Compute value for state.
        
        Args:
            instruction: Task instruction
            observation: Current state
            
        Returns:
            value: Scalar value estimate
        """
        # Format input
        text = f"Instruction: {instruction}\n\nState:\n{observation}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token or mean pooling
            hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Compute value
        value = self.value_head(hidden.to(torch.float32))
        
        return value.squeeze(-1)
    
    def forward_batch(
        self,
        instructions: list,
        observations: list
    ) -> torch.Tensor:
        """
        Batch value computation.
        
        Args:
            instructions: List of instructions
            observations: List of observations
            
        Returns:
            values: Tensor of shape [batch_size]
        """
        texts = [
            f"Instruction: {inst}\n\nState:\n{obs}"
            for inst, obs in zip(instructions, observations)
        ]
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            hidden = outputs.last_hidden_state[:, 0, :]
        
        # Compute values
        values = self.value_head(hidden.to(torch.float32))
        
        return values.squeeze(-1)