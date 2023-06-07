import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self, 
        input_size=576, 
        hidden_size=1024, 
        output_size=576, 
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class ResidualMLP(MLP):
    def __init__(self, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.residual_connection = nn.Linear(self.input_size, self.output_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x, **kwargs):
        residual = self.residual_connection(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x