import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

class SafetyPerceptionDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list or np.array): List or array of data samples.
            labels (list or np.array): List or array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
        self.img_path = "/data_nas/cehou/LLM_safety/PlacePulse2.0/photo_dataset/final_photo_dataset/"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.array(Image.open(f"{self.img_path}/{self.data[idx]['Image_ID']}.jpg"))
        image = Image.fromarray(image)
        label = self.data[idx]["labels"]
        if self.transform:            
            image = self.transform(image)

        return image, label
    
def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    

def create_dataset_from_df(data, with_nan=False):
    data_group = data.groupby("Image_ID")
    data_ls = []
    if with_nan:
        for i, (name, group) in tqdm(enumerate(data_group)):
            data_ls.append({
                "Image_ID": name,
                "labels": np.array(group['Q_Value'])
            })
    else:
        for i, (name, group) in tqdm(enumerate(data_group)):
            if not np.isnan(group['Q_Value']).any():
                data_ls.append({
                    "Image_ID": name,
                    "labels": np.array(group['Q_Value'])
                })
    return data_ls