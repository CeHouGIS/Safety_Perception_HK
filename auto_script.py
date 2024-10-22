import os
import neptune
import sys
import torch
from custom_clip_train import clip_train
sys.path.append('/code/LLM-crime/safety_perception_model/single_model')
from safety_train_withCLIP import safety_main, eval

import warnings
warnings.filterwarnings("ignore")

# 运行custom_clip_train.py
print("==============================================")
print("[1/3] Train CLIP model... Running custom_clip_train.py")
print("==============================================")

cfg_paras_clip = {
    'debug':False,
    'dataset_path':"/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_501.pkl",
    'save_model_path':"/data1/cehou_data/LLM_safety/LLM_model/clip_model",
    'save_model_name':"model_baseline_test.pt",
    'device':torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'batch_size':20,
    'num_workers':4,
    'head_lr':1e-3,
    'image_encoder_lr':1e-4,
    'text_encoder_lr':1e-5,
    'weight_decay':1e-3,
    'img_type':'PlacePulse',
    'patience':1,
    'factor':0.8,
    'epochs':400,
    'image_embedding':2048,
    'text_embedding':768,
    'max_length':512,
    'size':(300, 400),
    
    # models for image and text
    'model_name':'resnet50',
    'text_encoder_model':"distilbert-base-uncased",
    'text_tokenizer': "distilbert-base-uncased",
    'pretrained':True,
    'trainable':True,
    
    # deep learning model parameters
    'temperature':0.07,
    'projection_dim':256,
    'dropout':0.1,
    'early_stopping_threshold':20
    }

# cfg_paras_clip = {
# 'debug':False,
# 'save_model_path':"/data1/cehou_data/LLM_safety/LLM_model/clip_model",
# 'batch_size':20,
# 'head_lr':1e-3,
# 'image_encoder_lr':1e-4,
# 'text_encoder_lr':1e-5,
# 'weight_decay':1e-3,
# 'patience':1,
# 'factor':0.8,
# 'epochs':400,
# 'image_embedding':2048,
# 'text_embedding':768,
# 'max_length':512,
# 'size':(300, 400),

# # models for image and text
# 'model_name':'resnet50',
# 'text_encoder_model':"distilbert-base-uncased",
# 'text_tokenizer': "distilbert-base-uncased",

# # deep learning model parameters
# 'temperature':0.07,
# 'projection_dim':512,
# 'dropout':0.1,
# 'early_stopping_threshold':20
# }

clip_train(cfg_paras_clip)

# 运行safety_train.py
print("==============================================")
print("[2/3] Train deep learning model for safety perception evaluation...")
print("==============================================")

cfg_paras_safety = {
        'CLIP_model_path': "/data1/cehou_data/LLM_safety/LLM_model/clip_model/model_baseline_baseline_baseline_baseline.pt",
        'dataset_path': f"/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_501.pkl",
        'save_paths': f"/data1/cehou_data/LLM_safety/middle_variables/test",
        'model_save_path' : f"/data1/cehou_data/LLM_safety/safety_perception_model/",
        'placepulse_datapath': "/data_nas/cehou/LLM_safety/PlacePulse2.0/image_perception.csv",
        'eval_path': "/data1/cehou_data/LLM_safety/eval/",
        'train_type': 'classification',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'early_stopping_threshold':20
    }
safety_main(cfg_paras_safety)

# 评估结果，存入csv文件
# overall performance, confusion matrix, ROC curve, precision-recall curve
print("==============================================")
print("[3/3] Evaluate the models...")
print("==============================================")

eval(cfg_paras_safety)