# python /code/LLM-crime/safety_perception_model/single_model/safety_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from safety_perception_dataset import SafetyPerceptionCLIPDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import neptune
sys.path.append("/code/LLM-crime/safety_perception_model/single_model")
from my_models import FeatureViTClassifier, FeatureResNet50, LinearProbe
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

run = neptune.init_run(
    project="ce-hou/safety-llm",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
)  # your credentials

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

# 训练函数
def train_model(train_loader, valid_loader, paras):
    if paras['train_type'] == 'regression':
        output_dim = 2 
        # model = FeatureViTClassifier(output_dim).to(paras['device'])
        # model = FeatureResNet50(input_dim=256, num_classes=output_dim)
        model = LinearProbe(input_dim=256, num_classes=2).to(paras['device'])       
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=paras["CNN_lr"])
        
    elif paras['train_type'] == 'classification':
        # model = FeatureViTClassifier(output_dim=2).to(paras['device'])
        # model = FeatureResNet50(input_dim=256, num_classes=2).to(paras['device'])
        model = LinearProbe(input_dim=256, num_classes=2).to(paras['device'])        
        # class_weights = torch.tensor([1.0, 2.0, 3.0])  # 根据类别数量设置权重
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=paras["CNN_lr"])

    # Training loop
    num_epochs = paras['safety_epochs']
    best_loss = float('inf')
    count_after_best = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_running_loss = 0.0
        for inputs,labels in train_loader:
            inputs = inputs.to(paras['device'])
            labels = labels.to(paras['device']).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            # print("train_running_loss: ", loss.item())

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs,labels in valid_loader:
                inputs = inputs.to(paras['device'])
                labels = labels.to(paras['device']).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

        count_after_best += 1
        if val_running_loss < best_loss:
            best_loss = val_running_loss
            count_after_best = 0
            if not os.path.exists(paras['safety_model_save_path']):
                os.makedirs(paras['safety_model_save_path'])
            torch.save(model.state_dict(), os.path.join(paras['safety_model_save_path'], f"best_{paras['train_type']}_model.pth"))
            print(f"save the best model to {os.path.join(paras['safety_model_save_path'])}.")
        run["train/total_loss"].append(train_running_loss/len(train_loader))
        run["valid/total_loss"].append(val_running_loss/len(valid_loader))
        run["valid/accuracy"].append(correct / total)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/train_loader.batch_size:.4f}, Validation Loss: {val_running_loss/valid_loader.batch_size:.4f}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
        if count_after_best > paras['early_stopping_threshold']:
            print("Early Stopping!")
            break
        
def safety_main(paras):
    # 数据加载器
    img_feature,_ = get_img_feature(paras)
    print(img_feature.shape)
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
    train_dataset = SafetyPerceptionCLIPDataset(data_nonezero[:train_len].reset_index(), img_feature_nonezero[:train_len], paras)
    valid_dataset = SafetyPerceptionCLIPDataset(data_nonezero[train_len:].reset_index(), img_feature_nonezero[train_len:], paras)
    train_loader = DataLoader(train_dataset, batch_size=paras['batch_size_safety'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=paras['batch_size_safety'], shuffle=False)

    # 训练模型
    train_model(train_loader, valid_loader, paras)
    # train_model('regression', train_loader, valid_loader, device=device)
    
def eval(paras):
    # overall performance, confusion matrix, ROC curve, precision-recall curve
    # 后面只做一个专门用来validate的dataset
    img_feature = np.load(os.path.join(paras['variables_save_paths'], 'img_feature.npy'))  
    data = pd.read_csv(paras['placepulse_datapath'])
    SVI_namelist = pd.read_csv(paras['dataset_path'])
    namelist = pd.DataFrame([SVI_namelist.loc[i,'Image_ID'] for i in range(len(SVI_namelist))],columns=['Image_ID'])
    data = namelist.merge(data[data['Category'] == 'safety'], on='Image_ID')
    data_nonezero = data[data['label'] != 0]
    data_nonezero_idx = data[data['label'] != 0].index
    img_feature_nonezero = img_feature[data_nonezero_idx,:]
    data_nonezero = data_nonezero.reset_index(drop=True)
    data_nonezero.loc[data_nonezero[data_nonezero['label'] == -1].index,'label'] = 0



    valid_dataset = SafetyPerceptionCLIPDataset(data_nonezero, img_feature_nonezero, paras)
    valid_loader = DataLoader(valid_dataset, batch_size=paras['batch_size_safety'], shuffle=False)

    import matplotlib.pyplot as plt
    def evaluate_model(model, valid_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_preds

    # Load the best model
    # model = FeatureResNet50(input_dim=256, num_classes=2).to(paras['device'])
    model = LinearProbe(input_dim=256, num_classes=2).to(paras['device'])
    model.load_state_dict(torch.load(os.path.join(paras['safety_model_save_path'], f"best_{paras['train_type']}_model.pth")))
    all_labels, all_preds = evaluate_model(model, valid_loader, paras['device'])

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if not os.path.exists(os.path.join(paras['eval_path'], 'clip')):
        os.makedirs(os.path.join(paras['eval_path'], 'clip'))
    plt.savefig(os.path.join(paras['eval_path'], 'clip','test_confusion_matrix.png'))
    plt.close()
    
    # # Binarize the labels for ROC curve
    # all_labels_bin = label_binarize(all_labels, classes=list(range(2)))
    # all_preds_bin = label_binarize(all_preds, classes=list(range(2)))

    # Binarize the labels for ROC curve
    all_labels_bin = label_binarize(all_labels, classes=[0, 1])
    all_preds_bin = label_binarize(all_preds, classes=[0, 1])

    # Ensure the binarized labels have the correct shape
    if all_labels_bin.shape[1] == 1:
        all_labels_bin = np.hstack([1 - all_labels_bin, all_labels_bin])
    if all_preds_bin.shape[1] == 1:
        all_preds_bin = np.hstack([1 - all_preds_bin, all_preds_bin])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(14, 10))
    for i in range(2):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(paras['eval_path'], 'clip', 'test_roc_curve.png'))
    plt.close()
    # Compute Precision-Recall curve and area for each class

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(14, 10))
    for i in range(2):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(paras['eval_path'], 'clip', 'test_precision_recall_curve.png'))
    plt.close()