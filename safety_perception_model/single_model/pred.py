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
from safety_perception_model.single_model.my_models import TransformerRegressionModel
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = ['baseline','baseline','baseline','baseline']

input_paths = {
    "safety_model": "/data_nas/cehou/LLM_safety/PlacePulse2.0/model/safety_model.pth",
    "img_encoder": f"/data1/cehou_data/LLM_safety/LLM_model/clip_model/model_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}.pt",
    "dataset": f"/data1/cehou_data/LLM_safety/img_text_data/dataset_{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}_501.pkl"
}

save_paths = {
    "img_feature": f"/data1/cehou_data/LLM_safety/middle_variables/{cfg[0]}/img_feature.npy",
    "text_feature": f"/data1/cehou_data/LLM_safety/middle_variables/{cfg[0]}/text_feature.npy",
    "prediction": f"/data1/cehou_data/LLM_safety/middle_variables/{cfg[0]}/predictions.npy"
}

# Load the models
safety_model_path = input_paths['safety_model']
img_encoder_path = input_paths['img_encoder']
safety_model_paras = torch.load(safety_model_path)
img_encoder_paras = torch.load(img_encoder_path)

# Load the data
dataset_path = input_paths['dataset']
baseline_data = pd.read_pickle(dataset_path)

input_dim = 3
model_dim = 512  
num_heads = 8  
num_layers = 6  
output_dim = 6  

safety_model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim)
# safety_model = torch.nn.Sequential(*list(safety_model.children())[1:])
safety_model.load_state_dict(safety_model_paras, strict=False)
# print(safety_model)

cfg_paras = None

img_encoder = CLIPModel(cfg_paras)
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
img_feature, text_feature = make_prediction(img_encoder, train_loader) # (datasize, 256)
img_feature = np.array(img_feature)
text_feature = np.array(text_feature)

# Save img_feature to a file
img_feature_path = save_paths['img_feature']
np.save(img_feature_path, img_feature)
text_feature_path = save_paths['text_feature']
np.save(text_feature_path, text_feature)

# Transform (datasize, 256) to (datasize, 512)
linear_transform = torch.nn.Linear(256, 360000).to(device)
img_feature = torch.tensor(img_feature, dtype=torch.float32).to(device)
transformed_feature = linear_transform(img_feature)

# normalization
# mean = torch.tensor([0.485], device=device)
# std = torch.tensor([0.229], device=device)
# mean = mean.view(1, -1).expand_as(transformed_feature)
# std = std.view(1, -1).expand_as(transformed_feature)
# transformed_feature = (transformed_feature - mean) / std
# print(transformed_feature.shape)  # Should print (datasize, 256)
# predictions = safety_model(transformed_feature)

safety_model.eval()
with torch.no_grad():
    predictions = safety_model(transformed_feature)
    predictions_path = save_paths['prediction']
    np.save(predictions_path, predictions.cpu().numpy())
    print(predictions)
