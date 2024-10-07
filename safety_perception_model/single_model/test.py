# python /code/LLM-crime/safety_perception_model/single_model/test.py

import torch
import sys
sys.path.append("/code/LLM-crime")
from generate_features import generate_features

data_path = "/data_nas/cehou/LLM_safety/dataset_baseline_746.pkl"
model_path = "/data_nas/cehou/LLM_safety/model/model_baseline.pt"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
results = generate_features(data_path, model_path, device)

results.to_pickle("/data_nas/cehou/LLM_safety/img_features/features_baseline.pkl")