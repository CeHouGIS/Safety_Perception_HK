# python /code/LLM-crime/safety_perception_model/single_model/safety_train.py

import os
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
import neptune

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    


image_size = (300,400)
if os.path.exists("/data_nas/cehou/LLM_safety/PlacePulse2.0/train_data_ls.npy"):
    print("Loading data from file.")
    data_ls = np.load("/data_nas/cehou/LLM_safety/PlacePulse2.0/train_data_ls.npy", allow_pickle=True)
else:
    data = pd.read_csv("/data_nas/cehou/LLM_safety/PlacePulse2.0/image_perception.csv")
    data_ls = create_dataset_from_df(data, with_nan=False)

transform = get_transforms(image_size)
split_num = int(len(data_ls) * 0.8)
safety_dataset = SafetyPerceptionDataset(data_ls, transform=transform)
train_loader = torch.utils.data.DataLoader(safety_dataset[:split_num], batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(safety_dataset[split_num:], batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model, loss function, and optimizer
# Initialize the model, loss function, and optimizer
input_dim = 3
model_dim = 512  
num_heads = 8  
num_layers = 6  
output_dim = 6  


model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
# print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
best_loss = float('inf')
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float() # torch.Size([32, 3, 300, 400]), torch.Size([32, 6])

        optimizer.zero_grad()
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.item()
        print("Loss: ", train_loss.item())

    # valid
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            valid_loss = criterion(outputs, labels)

            val_running_loss += valid_loss.item()

    if val_running_loss < best_loss:
        best_loss = val_running_loss
        torch.save(model.state_dict(), "/data_nas/cehou/LLM_safety/PlacePulse2.0/model/safety_model.pth")
        print("save the best model.")
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_running_loss/len(valid_loader):.4f}")