import torch
import torch.nn as nn
import timm

# class TransformerRegressionModel(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
#         super(TransformerRegressionModel, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(model_dim, output_dim)
#         self.input_projection = nn.Linear(input_dim, model_dim)

#     def forward(self, x):
#         print("==============================")
#         x = self.input_projection(x)
#         print(x.shape)
#         x = self.transformer_encoder(x)
#         print(x.shape)
#         x = self.fc(x)
#         print(x.shape)
#         print("==============================")
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
        # print("==============================")
        batch_size = x.size(0) 
        x = x.view(batch_size, -1)  # Flatten the image tensor, torch.Size([32, 360000])
        # print(x.shape)
        x = self.input_projection(x) # torch.Size([32, 512])
        # print(x.shape)
        x = x.unsqueeze(0)  # Add sequence dimension for transformer, torch.Size([1, 32, 512])
        # print(x.shape)
        x = self.transformer(x, x)  # torch.Size([1, 32, 512])
        # print(x.shape)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_projection(x) # torch.Size([32, 6])
        # print(x.shape)
        # print("==============================")
        return x
    
    
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=20, input_dim=256):
        super(ViTClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if len(x.shape) > 1:
            x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x