# python /code/LLM-crime/auto_script.py
import os
import neptune
import sys
import torch
from custom_clip_train import clip_train
sys.path.append('/code/LLM-crime/safety_perception_model/single_model')
from safety_train_withCLIP import safety_main, eval
from generate_dataset import text_processing

import warnings
warnings.filterwarnings("ignore")

# 运行custom_clip_train.py

cfg_paras = {
    'debug':False,
    'dataset_path':"/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_1401.pkl",
    'save_model_path':"/data1/cehou_data/LLM_safety/LLM_models/clip_model/test",
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
    'early_stopping_threshold':20,
    
    # safety perception
    # 'CLIP_model_path': "/data1/cehou_data/LLM_safety/LLM_models/clip_model/test/model_baseline_best.pt",
    'variables_save_paths': f"/data1/cehou_data/LLM_safety/middle_variables/test",
    'safety_model_save_path' : f"/data1/cehou_data/LLM_safety/LLM_models/safety_perception_model/",
    'placepulse_datapath': "/data_nas/cehou/LLM_safety/PlacePulse2.0/image_perception.csv",
    'eval_path': "/data1/cehou_data/LLM_safety/eval/test/",
    'train_type': 'classification',
    'safety_epochs': 200,
    'CNN_lr': 1e-2,
    
    }


specific_paras = 'CNN_lr'
variable_paras = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for parameter in variable_paras:
    cfg_paras[specific_paras] = parameter
    print(f"specific_paras: {specific_paras}, parameter: {parameter}")
    cfg_paras['safety_model_save_path'] = f"/data1/cehou_data/LLM_safety/LLM_models/safety_perception_model/{specific_paras}_{parameter}/"
    cfg_paras['eval_path'] = f"/data1/cehou_data/LLM_safety/eval/{specific_paras}/{specific_paras}_{parameter}/"
    print("==============================================")
    print("[1/3] Train CLIP model... Running custom_clip_train.py")
    print("==============================================")

    # update data
    # text_processing(cfg_paras['dataset_path'], 'baseline')

    clip_train(cfg_paras)
    torch.cuda.empty_cache()

    # 运行safety_train.py
    print("==============================================")
    print("[2/3] Train deep learning model for safety perception evaluation...")
    print("==============================================")


    safety_main(cfg_paras)
    torch.cuda.empty_cache()

    # 评估结果，存入csv文件
    # overall performance, confusion matrix, ROC curve, precision-recall curve
    print("==============================================")
    print("[3/3] Evaluate the models...")
    print("==============================================")

    eval(cfg_paras)