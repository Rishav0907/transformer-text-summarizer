import torch
import torch.nn as nn
import torch.nn.init as init

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"

        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        self.W_O = nn.Linear(embedding_dim, embedding_dim)

        init.xavier_uniform_(self.W_K.weight)
        init.xavier_uniform_(self.W_Q.weight)
        init.xavier_uniform_(self.W_V.weight)
        init.xavier_uniform_(self.W_O.weight)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        K = self.W_K(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        Q = self.W_Q(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)

        output = self.W_O(context)

        return output

# Usage example:
# sa = SelfAttention(embedding_dim=512, num_heads=8)
# x = torch.rand(32, 10, 512)
# output = sa(x)