import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SubModules.decoder_block import DecoderBlock
from Modules.positional_encoding import PositionalEncoding

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class Decoder(nn.Module):
    def __init__(self, hidden_dims, num_layers):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding(hidden_dims)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(hidden_dims) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_dims, eps=1e-6)

    def forward(self, encoder_output, decoder_input):
        # Add positional encoding to the decoder input
        decoder_input = self.positional_encoding(decoder_input)
        # decoder_input = decoder_input + positional_encoding

        # Get the sequence length
        seq_len = decoder_input.size(1)

        # Create a causal mask for self-attention in the decoder to prevent looking ahead
        causal_mask = generate_square_subsequent_mask(seq_len).to(decoder_input.device)

        # Pass the decoder input through each decoder block
        for decoder_block in self.decoder_blocks:
            decoder_input = decoder_block(decoder_input, encoder_output, causal_mask)

        # Apply a final layer normalization after all decoder blocks
        decoder_output = self.layer_norm(decoder_input)

        # Handle NaN values in the final decoder output
        decoder_output = torch.where(torch.isnan(decoder_output), torch.zeros_like(decoder_output), decoder_output)

        return decoder_output
