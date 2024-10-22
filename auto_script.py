import os
import neptune
import sys
sys.path.append('/code/LLM-crime/safety_perception_model/single_model')
from custom_clip_train import clip_train

import warnings
warnings.filterwarnings("ignore")

# 运行custom_clip_train.py
print("==============================================")
print("[1/3] Train CLIP model... Running custom_clip_train.py")
print("==============================================")

cfg_paras_clip = {
'debug':False,
'save_model_path':"/data1/cehou_data/LLM_safety/LLM_model/clip_model",
'batch_size':20,
'head_lr':1e-3,
'image_encoder_lr':1e-4,
'text_encoder_lr':1e-5,
'weight_decay':1e-3,
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

# deep learning model parameters
'temperature':0.07,
'projection_dim':512,
'dropout':0.1,
'early_stopping_threshold':20
}

clip_train(cfg_paras_clip)

# 运行safety_train.py
print("==============================================")
print("[2/3] Train deep learning model for safety perception evaluation...")
print("==============================================")

cfg_paras_safety = {
    
}

# 评估结果，存入csv文件