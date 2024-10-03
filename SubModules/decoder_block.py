import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.multi_head_self_attention import MultiHeadSelfAttention
from Modules.feed_forward import FeedForwardNetwork
from Modules.positional_encoding import PositionalEncoding
from Modules.multi_head_cross_attention import MultiHeadCrossAttention

class DecoderBlock(nn.Module):
    def __init__(self, hidden_dims):
        super(DecoderBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.layer_norm1 = nn.LayerNorm(self.hidden_dims, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dims, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(self.hidden_dims, eps=1e-6)

        # Multi-head self-attention for the decoder
        self.multi_head_self_attention = MultiHeadSelfAttention(embedding_dim=self.hidden_dims, num_heads=8)

        # Multi-head cross-attention for encoder-decoder attention
        self.multi_head_cross_attention = MultiHeadCrossAttention(self.hidden_dims, num_heads=8)

        # Feed-forward network
        self.feed_forward_network = FeedForwardNetwork(input_dim=self.hidden_dims)

        # Positional encoding for the decoder
        self.positional_encoding = PositionalEncoding(self.hidden_dims)

    def forward(self, decoder_input, encoder_output, causal_mask):
        # Self-attention with causal mask (Decoder's look-ahead mechanism)
        self_attention_output = self.multi_head_self_attention(decoder_input, mask=causal_mask)
        self_attention_output = torch.where(torch.isnan(self_attention_output), torch.zeros_like(self_attention_output), self_attention_output)
        self_attention_output = self.layer_norm1(self_attention_output + decoder_input)

        # Cross-attention (Encoder-Decoder attention)
        cross_attention_output = self.multi_head_cross_attention(self_attention_output, encoder_output)
        cross_attention_output = torch.where(torch.isnan(cross_attention_output), torch.zeros_like(cross_attention_output), cross_attention_output)
        cross_attention_output = self.layer_norm2(cross_attention_output + self_attention_output)

        # Feed-forward network with residual connection
        feed_forward_output = self.feed_forward_network(cross_attention_output)
        feed_forward_output = torch.where(torch.isnan(feed_forward_output), torch.zeros_like(feed_forward_output), feed_forward_output)
        decoder_block_output = self.layer_norm3(feed_forward_output + cross_attention_output)

        return decoder_block_output
