import torch
import torch.nn as nn

# class TransformerRegressionModel(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
#         super(TransformerRegressionModel, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(model_dim, output_dim)
#         self.input_projection = nn.Linear(input_dim, model_dim)

#     def forward(self, x):
#         x = self.input_projection(x)
#         x = self.transformer_encoder(x)
#         x = self.fc(x)
#         return x
    
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerRegressionModel, self).__init__()
        self.input_projection = nn.Linear(input_dim * 300 * 400, model_dim)  # Adjust input_dim to match flattened image size
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the image tensor
        x = self.input_projection(x)
        x = x.unsqueeze(0)  # Add sequence dimension for transformer
        x = self.transformer(x, x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_projection(x)
        return x