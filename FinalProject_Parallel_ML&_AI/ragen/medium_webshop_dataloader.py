"""
Simple dataloader for Medium WebShop
"""
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class MediumWebShopDataset(Dataset):
    """Dataset for medium WebShop tasks."""

    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_medium_webshop_dataloaders(config: dict):
    """Create train/eval dataloaders for medium WebShop."""

    # Load datasets
    train_dataset = MediumWebShopDataset('data/medium_webshop_train.json')
    eval_dataset = MediumWebShopDataset('data/medium_webshop_eval.json')

    # Limit sizes based on config
    train_size = min(config['data']['train_size'], len(train_dataset))
    eval_size = min(config['data']['eval_size'], len(eval_dataset))

    print(f"Train samples: {train_size}")
    print(f"Eval samples: {eval_size}")

    # Create DataLoaders (batch_size=1 for RL)
    # Use identity collate_fn to return raw dicts, not batched tensors
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x  # Return list of dicts, not batched dict
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x  # Return list of dicts, not batched dict
    )

    return train_loader, eval_loader
