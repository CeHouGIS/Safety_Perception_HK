import torch
import torch.nn as nn
import torch.optim as optim
from models import TransformerRegressionModel
import sys
sys.path.append("/code/LLM-crime")
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from custom_clip_train import ImageEncoder

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    

def generate_pred_features(dataset, model, device):
    return None

# Hyperparameters
input_dim = 6
model_dim = 64
num_heads = 8
num_layers = 3
output_dim = 6
learning_rate = 0.001
num_epochs = 100

# Generate some dummy data
batch_size = 32
sequence_length = 10
x_train = torch.randn(batch_size, sequence_length, input_dim)
y_train = torch.randn(batch_size, sequence_length, output_dim)

# Initialize model, loss function, and optimizer
model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")