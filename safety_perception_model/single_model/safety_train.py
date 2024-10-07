# python /code/LLM-crime/safety_perception_model/single_model/safety_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from models import TransformerRegressionModel
import sys
sys.path.append("/code/LLM-crime/single_model")
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from safety_perception_dataset import *

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    


image_size = (300,400)
data = pd.read_csv("/data_nas/cehou/LLM_safety/PlacePulse2.0/image_perception.csv")
data_ls = create_dataset_from_df(data, with_nan=False)
transform = get_transforms(image_size)
safety_dataset = SafetyPerceptionDataset(data_ls, transform=transform)
data_loader = torch.utils.data.DataLoader(safety_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model, loss function, and optimizer
# Initialize the model, loss function, and optimizer
input_dim = 3  # Example value, replace with actual input dimension
model_dim = 512  # Example value, replace with actual model dimension
num_heads = 8  # Example value, replace with actual number of heads
num_layers = 6  # Example value, replace with actual number of layers
output_dim = 6  # Example value, replace with actual output dimension


model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        print(type(images), type(labels))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}")

print("Training complete.")