import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

class SafetyPerceptionDataset(Dataset):
    def __init__(self, data, transform=None, paras=None):
        """
        Args:
            data (list or np.array): List or array of data samples.
            labels (list or np.array): List or array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform
        self.img_path = "/data2/cehou/LLM_safety/PlacePulse2.0/photo_dataset/final_photo_dataset/"
        self.paras = paras

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = f"{self.img_path}/{self.data.iloc[idx]['Image_ID']}.jpg"
        image = np.array(Image.open(f"{self.img_path}/{self.data.iloc[idx]['Image_ID']}.jpg"))
        image = Image.fromarray(image)
        if self.paras['train_type'] == 'classification':
            label = self.data.iloc[idx]["label"]
            # label = label * 100 // 5
        elif self.paras['train_type'] == 'regression':
            label = self.data.iloc[idx]["Score"]
            
        if self.transform:            
            image = self.transform(image)

        return image, label
    
def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    

class SafetyPerceptionCLIPDataset(Dataset):
    def __init__(self, data, img_feature, paras):
        """
        Args:
            data (list or np.array): List or array of data samples.
            labels (list or np.array): List or array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = data
        self.image_feature = img_feature
        self.train_type = paras['train_type']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.image_feature.shape)
        image_feature_line = self.image_feature[idx,:]
        
        if self.train_type == 'classification':
            # label = self.data.iloc[idx,-1] * 100 // 5
            label = self.data.iloc[idx,'label']
        elif self.train_type == 'regression':
            label = self.data.iloc[idx,-1].astype(int)
        return image_feature_line, label
      

def create_dataset_from_df(data, with_nan=False, save=True):
    data_group = data.groupby("Image_ID")
    data_ls = []
    if with_nan:
        for i, (name, group) in tqdm(enumerate(data_group)):
            data_ls.append({
                "Image_ID": name,
                "labels": np.array(group['Score'])
            })
    else:
        for i, (name, group) in tqdm(enumerate(data_group)):
            if not np.isnan(group['Score']).any():
                data_ls.append({
                    "Image_ID": name,
                    "labels": np.array(group['Score'])
                })
    
    if save:
        np.save("/data2/cehou/LLM_safety/PlacePulse2.0/train_data_ls.npy", data_ls)
    return data_ls