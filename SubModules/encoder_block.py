import torch
import torch.nn as nn

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.multi_head_self_attention import MultiHeadSelfAttention
from Modules.feed_forward import FeedForwardNetwork
from configs.config import CONFIG

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dims):
        super(EncoderBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.multi_head_attention = MultiHeadSelfAttention(embedding_dim=self.hidden_dims, num_heads=8)
        self.feed_forward = FeedForwardNetwork(input_dim=self.hidden_dims)
        self.layer_norm1 = nn.LayerNorm(self.hidden_dims, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dims, eps=1e-6)

    def forward(self, input_token_tensor, mask=None):
        # Multi-Head Self-Attention with residual connection and layer norm
        attn_output = self.multi_head_attention(input_token_tensor, mask)
        residual1 = attn_output + input_token_tensor
        norm_output1 = self.layer_norm1(residual1)

        # Feed Forward Network with residual connection and layer norm
        feed_forward_output = self.feed_forward(norm_output1)
        residual2 = feed_forward_output + norm_output1
        norm_output2 = self.layer_norm2(residual2)

        return norm_output2


# Uncomment this block to test the EncoderBlock
# enc = EncoderBlock(64)
# data = torch.rand(32, 10, 64)
# output = enc(data)
# print(output.shape)
