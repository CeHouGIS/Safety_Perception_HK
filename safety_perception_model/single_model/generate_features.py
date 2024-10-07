import pandas as pd
import os
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append("/code/LLM-crime")
from custom_clip_train import ImageEncoder

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for i,path in enumerate(self.df[idx]["GSV_path"]):
            if i == 0:
                GSV_img = np.array(Image.open(path))
            else:
                GSV_img = np.concatenate((GSV_img, np.array(Image.open(path))), axis=1)
        if self.transform:
            images = self.transform(Image.fromarray(GSV_img))
            images = torch.tensor(images).permute(0, 1, 2).float()
        return images


# batch visualization
# import matplotlib.pyplot as plt
# # 获取一个批次的图像
# images = next(iter(data_loader))
# # 可视化图像
# fig, axes = plt.subplots(4, 8, figsize=(20, 10))
# axes = axes.flatten()
# for img, ax in zip(images, axes):
#     img = img.permute(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
#     img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # 反归一化
#     img = img.numpy()
#     ax.imshow(img)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

def get_features(data_loader, model, state_dict, device):
    # Remove the prefix 'image_encoder.model.' from the state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('image_encoder.model.'):
            new_state_dict[k[len('image_encoder.model.'):]] = v
        else:
            new_state_dict[k] = v

    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict, strict=False).to(device)
    # Set the model to evaluation mode
    model.eval()
    # Get the results
    results = []
    with torch.no_grad():
        for batch in data_loader:
            batch_results = model(batch)
            results.append(batch_results)

    # Concatenate all results
    results = torch.cat(results, dim=0)
    return results

def generate_features(data_path, model_path, device):
    print(f"device: {device}")

    df = pd.read_pickle(data_path)

    model = ImageEncoder()
    if device == 'cpu':
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(model_path)
        
    transform = get_transforms((int(320/2), int(1280/2)))
    image_dataset = ImageDataset(df, transform=transform)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
    print(f"device: {device}")
    result = get_features(data_loader, model, state_dict, device)
    return result