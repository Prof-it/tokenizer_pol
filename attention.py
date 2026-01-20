import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_size, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        # Single linear projection for all Q, K, V across all heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Register mask once
        self.register_buffer('mask', torch.triu(
            torch.ones(context_size, context_size, dtype=torch.bool),
            diagonal=1
        ))

    def forward(self, x):
        B, T, C = x.shape

        # Single matrix multiplication for all heads' Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, T, d_head)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        # Apply causal mask
        scores.masked_fill_(self.mask[:T, :T], float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = attn_weights @ v  # (B, num_heads, T, d_head)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        output = self.out_proj(context)
        return self.dropout(output)