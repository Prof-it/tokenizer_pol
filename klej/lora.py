"""LoRA (Low-Rank Adaptation) implementation for custom transformer."""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    LoRA adapter for nn.Linear layers.

    Adds low-rank decomposition: W' = W + BA where B is (out, r) and A is (r, in).
    Only A and B are trainable; original W is frozen.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_layer(x)

        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: list = None,
) -> nn.Module:
    """
    Apply LoRA to specified modules in the model.

    Args:
        model: The model to modify
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor
        dropout: Dropout for LoRA layers
        target_modules: List of module name patterns to apply LoRA to.
                       Default: ['qkv_proj', 'out_proj'] for attention layers

    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ['qkv_proj', 'out_proj']

    # Find and replace target modules
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)

    return model


def get_lora_params(model: nn.Module) -> list:
    """Get only LoRA parameters for optimizer."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, and LoRA parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)

    return {
        'total': total,
        'trainable': trainable,
        'lora': lora,
        'frozen': total - trainable,
    }