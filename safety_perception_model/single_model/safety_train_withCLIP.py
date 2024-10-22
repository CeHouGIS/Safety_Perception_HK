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
from safety_perception_dataset import SafetyPerceptionCLIPDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import neptune
sys.path.append("/code/LLM-crime/safety_perception_model/single_model")
from models import TransformerRegressionModel
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

run = neptune.init_run(
    project="ce-hou/Safety",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
)  # your credentials

def train_safety_model():
    CLIP_model_path = "/data1/cehou_data/LLM_safety/LLM_model/clip_model/model_baseline_baseline_baseline_baseline.pt"
    dataset_path = f"/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_501.pkl"
    save_paths = f"/data1/cehou_data/LLM_safety/middle_variables/test"
    if not os.path.exists(save_paths):
        os.makedirs(save_paths)
    text_tokenizer = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_encoder_paras = torch.load(CLIP_model_path)
    img_encoder = CLIPModel()
    img_encoder.load_state_dict(img_encoder_paras)
    baseline_data = pd.read_pickle(dataset_path)
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)

    train_num = int(len(baseline_data) * 0.7)
    train_loader = build_loaders(baseline_data[:train_num], tokenizer, mode="train")
    valid_loader = build_loaders(baseline_data[train_num:], tokenizer, mode="valid")

    img_encoder.to(device)
    img_feature, text_feature = make_prediction(img_encoder, train_loader) # (datasize, 256)
    img_feature = np.array(img_feature)
    text_feature = np.array(text_feature)

    # Save img_feature to a file
    np.save(os.path.join(save_paths, 'img_feature.npy'), img_feature)
    np.save(os.path.join(save_paths, 'text_feature.npy'), text_feature)

    # Initialize the model, loss function, and optimizer
    input_dim = 3
    model_dim = 2048
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
        train_running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float() # torch.Size([32, 3, 300, 400]), torch.Size([32, 6])

            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            run["train/loss"].append(train_loss)

            train_running_loss += train_loss.item()

        # valid
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                valid_loss = criterion(outputs, labels)
                run["valid/loss"].append(valid_loss)
                val_running_loss += valid_loss.item()

        if val_running_loss < best_loss:
            best_loss = val_running_loss
            torch.save(model.state_dict(), "/data_nas/cehou/LLM_safety/PlacePulse2.0/model/safety_model.pth")
            print("save the best model.")
        
        run["train/total_loss"].append(train_running_loss/len(train_loader))
        run["valid/total_loss"].append(val_running_loss/len(valid_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/len(train_loader):.4f}, Validation Loss: {val_running_loss/len(valid_loader):.4f}")
