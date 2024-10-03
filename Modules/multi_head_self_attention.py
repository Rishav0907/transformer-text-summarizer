import torch
import torch.nn as nn
import torch.nn.init as init
from Modules.self_attention import SelfAttention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.attention_heads = nn.ModuleList([
            SelfAttention(embedding_dim=self.head_dim, num_heads=1) for _ in range(self.num_heads)
        ])

        self.W_O = nn.Linear(embedding_dim, embedding_dim)
        init.xavier_uniform_(self.W_O.weight)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # Split input for each head
        split_x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Process each head
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_input = split_x[:, :, i, :].contiguous()
            head_output = head(head_input, mask)
            head_outputs.append(head_output)

        # Concatenate outputs from all heads
        multi_head_output = torch.cat(head_outputs, dim=-1)

        # Apply output projection
        output = self.W_O(multi_head_output)

        return output

# Usage example:
# mhsa = MultiHeadSelfAttention(embedding_dim=512, num_heads=8)
# x = torch.rand(32, 10, 512)
# output = mhsa(x)