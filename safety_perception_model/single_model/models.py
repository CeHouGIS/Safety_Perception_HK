import torch
import torch.nn as nn

class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerRegressionModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x