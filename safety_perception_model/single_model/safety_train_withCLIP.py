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
from models import TransformerRegressionModel, ViTClassifier
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

run = neptune.init_run(
    project="ce-hou/Safety",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
)  # your credentials


def get_img_feature(paras):
    CLIP_model_path = paras['CLIP_model_path']
    dataset_path = paras['dataset_path']
    save_paths = paras['save_paths']
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
    return img_feature, text_feature

# 训练函数
def train_model(train_loader, valid_loader, paras):
    if paras['train_type'] == 'regression':
        input_dim = 3
        model_dim = 2048
        num_heads = 8  
        num_layers = 6  
        output_dim = 6  

        model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(paras['device'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    elif paras['train_type'] == 'classification':
        model = ViTClassifier(num_classes=20).to(paras['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    count_after_best = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_running_loss = 0.0
        for batch in train_loader:
            inputs = batch['input'].to(paras['device'])
            labels = batch['label'].to(paras['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(paras['device'])
                labels = batch['label'].to(paras['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

        count_after_best += 1
        if val_running_loss < best_loss:
            best_loss = val_running_loss
            count_after_best = 0
            torch.save(model.state_dict(), os.path.join(paras['model_save_path'], f"best_{paras['train_type']}_model.pth"))
            print(f"save the best model to {os.path.join(paras['model_save_path'])}.")

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/len(train_loader):.4f}, Validation Loss: {val_running_loss/len(valid_loader):.4f}")      
        run["train/total_loss"].append(train_running_loss/len(train_loader))
        run["valid/total_loss"].append(val_running_loss/len(valid_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/len(train_loader):.4f}, Validation Loss: {val_running_loss/len(valid_loader):.4f}")
        if count_after_best > paras['early_stopping_threshold']:
            print("Early Stopping!")
            break
        
def safety_main(paras):
    # 数据加载器
    img_feature = get_img_feature(paras)
    data = pd.read_csv(paras['placepulse_datapath'])
    train_len = 0.7*len(img_feature)
    valid_len = len(img_feature) - 0.7*len(img_feature)
    train_dataset = SafetyPerceptionCLIPDataset(data[:train_len], img_feature[:train_len])
    valid_dataset = SafetyPerceptionCLIPDataset(data[train_len:valid_len], img_feature[train_len:valid_len])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 训练模型
    train_model(train_loader, valid_loader, paras)
    # train_model('regression', train_loader, valid_loader, device=device)
    
def eval(paras):
    # overall performance, confusion matrix, ROC curve, precision-recall curve
    # 后面只做一个专门用来validate的dataset
    img_feature = np.load(os.path.join(paras['save_paths'], 'img_feature.npy'))
    data = pd.read_csv(paras['placepulse_datapath'])
    valid_dataset = SafetyPerceptionCLIPDataset(data, img_feature)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    import matplotlib.pyplot as plt

    def evaluate_model(model, valid_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['input'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_preds

    # Load the best model
    model = ViTClassifier(num_classes=20).to(paras['device'])
    model.load_state_dict(torch.load(os.path.join(paras['model_save_path'], f"best_{paras['train_type']}_model.pth")))
    all_labels, all_preds = evaluate_model(model, valid_loader, paras['device'])

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(paras['model_save_path'], os.path.join(paras['eval_path'],'test_confusion_matrix.png')))

    # Binarize the labels for ROC curve
    all_labels_bin = label_binarize(all_labels, classes=list(range(20)))
    all_preds_bin = label_binarize(all_preds, classes=list(range(20)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(20):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(14, 10))
    for i in range(20):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(paras['model_save_path'], os.path.join(paras['eval_path'], 'test_roc_curve.png')))
    
    # Compute Precision-Recall curve and area for each class

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(20):
        precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_preds_bin[:, i])
        average_precision[i] = average_precision_score(all_labels_bin[:, i], all_preds_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(14, 10))
    for i in range(20):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(paras['model_save_path'], os.path.join(paras['eval_path'], 'test_precision_recall_curve.png')))