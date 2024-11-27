# python /code/LLM-crime/auto_script.py
# cp -rv /data2/cehou/LLM_safety/PlacePulse2.0/* /data2/cehou/LLM_safety/PlacePulse2.0/

import os
import neptune
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from custom_clip_train import clip_train
sys.path.append('/code/LLM-crime/safety_perception_model/single_model')
from safety_train_withCLIP import safety_main, eval
from generate_dataset import text_processing
from safety_perception_dataset import SafetyPerceptionCLIPDataset
from my_models import TransformerRegressionModel, FeatureViTClassifier
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# 运行custom_clip_train.py

def get_img_feature(paras):
    CLIP_model_path = os.path.join(paras['save_model_path'], paras['save_model_name'])
    save_paths = paras['variables_save_paths']
    if not os.path.exists(save_paths):
        os.makedirs(save_paths)
    text_tokenizer = "distilbert-base-uncased"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {paras['device']}")

    img_encoder_paras = torch.load(CLIP_model_path)
    img_encoder = CLIPModel(paras)
    img_encoder.load_state_dict(img_encoder_paras)
    baseline_data = pd.read_csv(paras['dataset_path'])
    # print("baseline_data: ", len(baseline_data))
    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)

    data_loader = build_loaders(baseline_data, tokenizer, mode="valid", cfg_paras=paras)

    img_encoder.to(paras['device'])
    img_feature, text_feature = make_prediction(img_encoder, data_loader, cfg_paras=paras) # (datasize, 256)
    img_feature = np.array(img_feature)
    text_feature = np.array(text_feature)

    # Save img_feature to a file
    np.save(os.path.join(save_paths, 'img_feature.npy'), img_feature)
    np.save(os.path.join(save_paths, 'text_feature.npy'), text_feature)
    return img_feature, text_feature

def ml_eval(paras):
    img_feature,text_feature = get_img_feature(paras)
    np.save(os.path.join(paras['eval_path'], 'img_feature.npy'), img_feature)
    np.save(os.path.join(paras['eval_path'], 'text_feature.npy'), text_feature)
    
    data = pd.read_csv(paras['placepulse_datapath'])
    SVI_namelist = pd.read_csv(paras['dataset_path'])
    namelist = pd.DataFrame([SVI_namelist.loc[i,'Image_ID'] for i in range(len(SVI_namelist))],columns=['Image_ID'])
    data = namelist.merge(data[data['Category'] == 'safety'], on='Image_ID')
    data_nonezero = data[data['label'] != 0]
    data_nonezero_idx = data[data['label'] != 0].index
    img_feature_nonezero = img_feature[data_nonezero_idx,:]
    data_nonezero = data_nonezero.reset_index(drop=True)
    data_nonezero.loc[data_nonezero[data_nonezero['label'] == -1].index,'label'] = 0

    train_len = int(0.7*len(img_feature_nonezero))
    train_dataset = SafetyPerceptionCLIPDataset(data[:train_len], img_feature[:train_len], paras)
    valid_dataset = SafetyPerceptionCLIPDataset(data[train_len:], img_feature[train_len:], paras)
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    # 准备训练和验证数据
    train_len = int(0.5*len(img_feature_nonezero))
    X_train = img_feature_nonezero[:train_len]
    y_train = data_nonezero['label'][:train_len]
    X_valid = img_feature_nonezero[train_len:]
    y_valid = data_nonezero['label'][train_len:]

    # 初始化随机森林分类器
    if paras['ml_model'] == 'RandomForest':
        rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_valid)

    # 打印分类报告和准确率
    print("Classification Report:\n", classification_report(y_valid, y_pred))
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_valid, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(paras['eval_path'], 'confusion_matrix.png'))
    plt.clf()
    return accuracy_score(y_valid, y_pred)
    
def main():
    
    cfg_paras = {
    'debug':False,
    # 'dataset_path':"/data2/cehou/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_1401.pkl",
    'dataset_path':'/data2/cehou/LLM_safety/img_text_data/baseline/tidyed/dataset_baseline_baseline_baseline_baseline_9030_withlabel.csv',
    'save_model_path':"/data_nas/cehou/LLM_safety/LLM_models/clip_model/test",
    'save_model_name':"model_baseline_test.pt",
    'device':torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    'CLIP_train_type': 'train', # train, finetune
    'batch_size':60,
    'num_workers':4,
    'head_lr':1e-3,
    'temperature':0.05,
    'image_encoder_lr':0.000100,
    'text_encoder_lr':0.000045,
    'weight_decay':1e-3,
    'img_type':'PlacePulse',
    'patience':1,
    'factor':0.8,
    'epochs':999,
    'image_embedding':768,
    'text_embedding':768,
    'max_length':512,
    'size':(112,112),
    
    # models for image and text
    'ml_model':'RandomForest',
    'model_name':'resnet50',
    'text_encoder_model':"distilbert-base-uncased",
    'text_tokenizer': "distilbert-base-uncased",
    'pretrained':True,
    'trainable':True,
    
    # deep learning model parameters
    'projection_dim':256,
    'dropout':0.1,
    'early_stopping_threshold':5,
    
    # safety perception
    # 'CLIP_model_path': "/data2/cehou/LLM_safety/LLM_models/clip_model/test/model_baseline_best.pt",
    'variables_save_paths': f"/data2/cehou/LLM_safety/middle_variables/test",
    'safety_model_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/",
    'placepulse_datapath': "/data2/cehou/LLM_safety/PlacePulse2.0/image_perception_score.csv",
    'eval_path': "/data2/cehou/LLM_safety/eval/test/",
    'train_type': 'classification',
    'safety_epochs': 200,
    'batch_size_safety': 256,
    'CNN_lr': 1e-2,
    }
    
    # single loop
    specific_paras = ['temperature']
    variable_paras = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for i in variable_paras:
        cfg_paras['device'] = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        cfg_paras[specific_paras[0]] = i
        save_folder_name = '_'.join(specific_paras)
        save_paras = [specific_paras[0], i]
        save_file_name = '_'.join([str(x) for x in save_paras])
        cfg_paras['save_model_path'] = f"/data_nas/cehou/LLM_safety/LLM_models/clip_model/{save_folder_name}/{save_file_name}"
        if not os.path.exists(cfg_paras['save_model_path']):
            os.makedirs(cfg_paras['save_model_path'])
        cfg_paras['save_model_name'] = f"model_baseline_{save_file_name}.pt"
        cfg_paras['safety_model_save_path'] = f"/data_nas/cehou/LLM_safety/LLM_models/safety_perception_model/{save_file_name}/"
        cfg_paras['eval_path'] = f"/data2/cehou/LLM_safety/eval/{save_folder_name}/{save_file_name}/"
        if not os.path.exists(cfg_paras['safety_model_save_path']):
            os.makedirs(cfg_paras['safety_model_save_path'])
        if not os.path.exists(cfg_paras['eval_path']):
            os.makedirs(cfg_paras['eval_path'])

        print("==============================================")
        print("[1/2] Train CLIP model... Running custom_clip_train.py")
        print("==============================================")

        clip_train(cfg_paras)
        torch.cuda.empty_cache()
        
        print("==============================================")
        print("[2/2] safety perception model... Running safety_train_withCLIP.py")
        print("==============================================")
        
        accuracy = ml_eval(cfg_paras)
        cfg_paras['device'] = 'cuda:3'
        cfg_paras['accuracy'] = accuracy
        pd.DataFrame(cfg_paras).to_csv(os.path.join(cfg_paras['eval_path'], 'cfg_paras.csv'))
    
    # double loop
    specific_paras = ['image_encoder_lr', 'text_encoder_lr']
    variable_paras = np.linspace(1e-4, 1e-6, 10)
    # variable_paras = [1e-2]

    
    
    # for i in variable_paras:
    #     for j in variable_paras:
    #         cfg_paras['device'] = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #         cfg_paras[specific_paras[0]] = i
    #         cfg_paras[specific_paras[1]] = j
    #         save_folder_name = '_'.join(specific_paras)
    #         save_paras = [specific_paras[0], i, specific_paras[1], j]
    #         save_file_name = '_'.join([str(x) for x in save_paras])
    #         cfg_paras['save_model_path'] = f"/data_nas/cehou/LLM_safety/LLM_models/clip_model/{save_folder_name}/{save_file_name}"
    #         if not os.path.exists(cfg_paras['save_model_path']):
    #             os.makedirs(cfg_paras['save_model_path'])
    #         cfg_paras['save_model_name'] = f"model_baseline_{save_file_name}.pt"
    #         cfg_paras['safety_model_save_path'] = f"/data_nas/cehou/LLM_safety/LLM_models/safety_perception_model/{save_file_name}/"
    #         cfg_paras['eval_path'] = f"/data2/cehou/LLM_safety/eval/{save_folder_name}/{save_file_name}/"
    #         if not os.path.exists(cfg_paras['safety_model_save_path']):
    #             os.makedirs(cfg_paras['safety_model_save_path'])
    #         if not os.path.exists(cfg_paras['eval_path']):
    #             os.makedirs(cfg_paras['eval_path'])

    #         print("==============================================")
    #         print("[1/2] Train CLIP model... Running custom_clip_train.py")
    #         print("==============================================")

    #         clip_train(cfg_paras)
    #         torch.cuda.empty_cache()
            
    #         print("==============================================")
    #         print("[2/2] safety perception model... Running safety_train_withCLIP.py")
    #         print("==============================================")
            
    #         accuracy = ml_eval(cfg_paras)
    #         cfg_paras['device'] = 'cuda:3'
    #         cfg_paras['accuracy'] = accuracy
    #         pd.DataFrame(cfg_paras).to_csv(os.path.join(cfg_paras['eval_path'], 'cfg_paras.csv'))
            
        
if __name__ == '__main__':
    main()
        
# for parameter in variable_paras:
#     cfg_paras[specific_paras] = parameter
#     print(f"specific_paras: {specific_paras}, parameter: {parameter}")
#     save_filename = '_'.join(specific_paras)
#     cfg_paras['safety_model_save_path'] = f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/{save_filename}_{parameter}/"
#     cfg_paras['eval_path'] = f"/data2/cehou/LLM_safety/eval/{save_filename}/{specific_paras}_{parameter}/"
#     print("==============================================")
#     print("[1/4] Train CLIP model... Running custom_clip_train.py")
#     print("==============================================")

    # update data
    # text_processing(cfg_paras['dataset_path'], 'baseline')

    # clip_train(cfg_paras)
    # torch.cuda.empty_cache()
    

    # 运行safety_train.py
    # print("==============================================")
    # print("[2/4] Finetune CLIP model... Running custom_clip_train.py")
    # print("==============================================")
    
    # finetune CLIP model
    # cfg_paras['CLIP_train_type'] = 'finetune'
    # clip_train(cfg_paras)
    # torch.cuda.empty_cache()

    # 运行safety_train.py
    # print("==============================================")
    # print("[3/4] Train deep learning model for safety perception evaluation...")
    # print("==============================================")


    # safety_main(cfg_paras)
    # torch.cuda.empty_cache()

    # 评估结果，存入csv文件
    # overall performance, confusion matrix, ROC curve, precision-recall curve
    # print("==============================================")
    # print("[4/4] Evaluate the models...")
    # print("==============================================")

    # eval(cfg_paras)