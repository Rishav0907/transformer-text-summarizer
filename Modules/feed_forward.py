import torch
import torch.nn as nn
import torch.nn.init as init

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or 4 * input_dim
        self.output_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Usage example:
# ffn = FeedForwardNetwork(input_dim=512, hidden_dim=2048, dropout_rate=0.1)
# x = torch.rand(32, 10, 512)
# output = ffn(x)