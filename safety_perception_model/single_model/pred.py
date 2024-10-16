# python /code/LLM-crime/safety_perception_model/single_model/pred.py
import torch
import pandas as pd
import os
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
import sys
sys.path.append("/code/LLM-crime/safety_perception_model/single_model")
from models import TransformerRegressionModel
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

# Load the models
safety_model_path = "/data_nas/cehou/LLM_safety/PlacePulse2.0/model/safety_model.pth"
img_encoder_path = "/data_nas/cehou/LLM_safety/model/model_baseline.pt"
safety_model_paras = torch.load(safety_model_path)
img_encoder_paras = torch.load(img_encoder_path)

# Load the data
dataset_path = "/data_nas/cehou/LLM_safety/dataset_baseline_746.pkl"
baseline_data = pd.read_pickle(dataset_path)

input_dim = 3
model_dim = 512  
num_heads = 8  
num_layers = 6  
output_dim = 6  

safety_model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim)
safety_model.load_state_dict(safety_model_paras, strict=False)

img_encoder = CLIPModel()
img_encoder.load_state_dict(img_encoder_paras)
text_tokenizer = "distilbert-base-uncased"
train_num = int(len(baseline_data) * 0.7)
tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)

train_loader = build_loaders(baseline_data[:train_num], tokenizer, mode="train")
valid_loader = build_loaders(baseline_data[train_num:], tokenizer, mode="valid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

safety_model.to(device)
img_encoder.to(device)
output = make_prediction(img_encoder, valid_loader)
print(output)