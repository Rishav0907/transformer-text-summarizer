import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SubModules.encoder_block import EncoderBlock
from Modules.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, hidden_dims, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.positional_encoding = PositionalEncoding(hidden_dims)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(hidden_dims) for _ in range(num_layers)])

    def forward(self, input_token_tensor, mask=None):
        # input_token_tensor shape: (batch_size, sequence_length, hidden_dims)
        batch_size, seq_len, _ = input_token_tensor.size()

        # Positional Encoding
        input_token_tensor = self.positional_encoding(input_token_tensor)
        # input_token_tensor = input_token_tensor + positional_encoding

        # Passing through each encoder block
        for encoder_block in self.encoder_blocks:
            input_token_tensor = encoder_block(input_token_tensor, mask)

        return input_token_tensor

