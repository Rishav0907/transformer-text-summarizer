import torch
import torch.nn as nn
import torch.nn.init as init
from Modules.cross_attention import CrossAttention

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.attention_heads = nn.ModuleList([
            CrossAttention(embedding_dim=self.head_dim, num_heads=1) for _ in range(self.num_heads)
        ])

        self.W_O = nn.Linear(embedding_dim, embedding_dim)
        init.xavier_uniform_(self.W_O.weight)

    def forward(self, query, key_value, mask=None):
        batch_size, query_len, _ = query.size()
        _, key_value_len, _ = key_value.size()

        # Split input for each head
        split_query = query.view(batch_size, query_len, self.num_heads, self.head_dim)
        split_key_value = key_value.view(batch_size, key_value_len, self.num_heads, self.head_dim)

        # Process each head
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_query = split_query[:, :, i, :].contiguous()
            head_key_value = split_key_value[:, :, i, :].contiguous()
            head_output = head(head_query, head_key_value, mask)
            head_outputs.append(head_output)

        # Concatenate outputs from all heads
        multi_head_output = torch.cat(head_outputs, dim=-1)

        # Apply output projection
        output = self.W_O(multi_head_output)

        return output

# Usage example:
# mhca = MultiHeadCrossAttention(embedding_dim=512, num_heads=8)
# query = torch.rand(32, 10, 512)  # decoder output
# key_value = torch.rand(32, 20, 512)  # encoder output
# output = mhca(query, key_value)