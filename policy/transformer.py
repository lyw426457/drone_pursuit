import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=256, num_encoder_layers=4)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)
