# python /code/LLM-crime/safety_perception_model/single_model/safety_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("/code/LLM-crime/single_model")
from my_models import TransformerRegressionModel, ViTClassifier, ResNet50Model
from PIL import Image
import torchvision.transforms as transforms
from safety_perception_dataset import *
import neptune
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import r2_score
import shutil

# CUDA_LAUNCH_BLOCKING=1

run = neptune.init_run(
    project="ce-hou/Safety",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
)  # your credentials

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )    

def train_model(train_loader, valid_loader, paras):
    print(f'device: {paras["device"]}')
    if paras['train_type'] == 'regression':
        input_dim = 3 * paras['size'][0] * paras['size'][1]
        model_dim = 512
        num_heads = 8  
        num_layers = 6
        dropout = paras['dropout']
        output_dim = 1
        model = ResNet50Model(output_dim).to(paras['device'])
        # print(model)
        # model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout).to(paras['device'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=paras["CNN_lr"])
        
    elif paras['train_type'] == 'classification':
        input_dim = 3 * paras['size'][0] * paras['size'][1]
        output_dim = paras['class_num']
        model = ResNet50Model(output_dim).to(paras['device'])
        # print(model)
        # model = ViTClassifier(num_classes=paras['class_num'],input_dim=input_dim).to(paras['device'])
        if paras['weight_on']:
            class_weights = torch.FloatTensor(paras['class_weights']).to(paras['device'])
            # print("class_weights: ", class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=paras["CNN_lr"])

    # Training loop
    num_epochs = paras['safety_epochs']
    best_loss = float('inf')
    count_after_best = 0
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        tqdm_loader = tqdm(train_loader, total=len(train_loader))
        for inputs,labels in tqdm_loader:
            inputs = inputs.to(paras['device']) #16, 3, 300, 400
            # print(labels)
            if paras['train_type'] == 'classification':
                labels = labels.to(paras['device']).long()
            elif paras['train_type'] == 'regression':
                labels = labels.to(paras['device']).float()
                
            optimizer.zero_grad()
            outputs = model(inputs) # 16, 6
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            # print("train_running_loss: ", loss.item())
            
            # Update tqdm description with current loss
            tqdm_loader.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            tqdm_loader.set_postfix(loss=train_running_loss)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs,labels) in enumerate(valid_loader):
                if paras['train_type'] == 'classification':
                    inputs = inputs.to(paras['device'])
                    labels = labels.to(paras['device']).long()
                    # print(labels)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Record predictions and true labels
                    if i == 0:
                        all_preds = predicted.cpu().numpy()
                        all_labels = labels.cpu().numpy()
                    else:
                        all_preds = np.concatenate((all_preds, predicted.cpu().numpy()))
                        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))
                        
                elif paras['train_type'] == 'regression':
                    inputs = inputs.to(paras['device'])
                    labels = labels.to(paras['device']).float()
                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    if i == 0:
                        all_preds = outputs.cpu().numpy()
                        all_labels = labels.cpu().numpy()
                    else:
                        all_preds = np.concatenate((all_preds, outputs.cpu().numpy()))
                        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/train_loader.batch_size:.4f}, Validation Loss: {val_running_loss/valid_loader.batch_size:.4f}")
        if paras['train_type'] == 'classification':
            run["valid/accuracy"].append(correct / total)
            print(f"Accuracy: {100 * correct / total:.2f}%")
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                        annot_kws={"size": 12, "weight": "bold", "color": "red"}, 
                        mask=np.eye(cm.shape[0], dtype=bool))
            plt.title(f"Confusion Matrix epoch {epoch+1} acc: {correct/total:0.2%}")
            cm_savepath = os.path.join(paras['eval_path'], 'valid_cm')
            if not os.path.exists(cm_savepath):
                os.makedirs(cm_savepath)
            plt.savefig(os.path.join(cm_savepath, f"confusion_matrix_epoch_{epoch+1}.png"))
            plt.close()
        elif paras['train_type'] == 'regression':
            r2 = r2_score(all_preds, all_labels)
            run["valid/r2_score"].append(r2)
            print(f"R2 score: {r2:.2f}")       
            # Plot R2 score curve
            plt.figure(figsize=(10, 8))
            sns.regplot(x=all_labels, y=all_preds, scatter_kws={'s':10}, line_kws={"color":"red"})
            plt.xlim(-0.5,1.5)
            plt.ylim(-0.5,1.5)
            plt.xlabel("True Labels")
            plt.ylabel("Predicted Labels")
            plt.title(f"Regression Results epoch {epoch+1} R2: {r2:.2f}")
            regplot_savepath = os.path.join(paras['eval_path'], 'regression_plots')
            if not os.path.exists(regplot_savepath):
                os.makedirs(regplot_savepath)
            plt.savefig(os.path.join(regplot_savepath, f"regression_plot_epoch_{epoch+1}.png"))
            plt.close()
                
        if count_after_best > paras['early_stopping_threshold']:
            print("Early Stopping!")
            break

# if os.path.exists("/data2/cehou/LLM_safety/PlacePulse2.0/train_data_ls.npy"):
#     print("Loading data from file.")
#     data_ls = np.load("/data2/cehou/LLM_safety/PlacePulse2.0/train_data_ls.npy", allow_pickle=True)
# else:
#     data = pd.read_csv("/data2/cehou/LLM_safety/PlacePulse2.0/image_perception.csv")
#     data_ls = create_dataset_from_df(data, with_nan=False)

cfg_paras = {
    'debug':False,
    'dataset_path':"/data2/cehou/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_1401.pkl",
    'save_model_path':"/data2/cehou/LLM_safety/LLM_models/clip_model/test",
    'save_model_name':"model_baseline_test.pt",
    'device':torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    'batch_size':256,
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
    'size':(224,224),
    
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
    'early_stopping_threshold':5,
    
    # safety perception
    # 'CLIP_model_path': "/data2/cehou/LLM_safety/LLM_models/clip_model/test/model_baseline_best.pt",
    'variables_save_paths': f"/data2/cehou/LLM_safety/middle_variables/test",
    'safety_model_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/only_img/",
    'placepulse_datapath': "/data2/cehou/LLM_safety/PlacePulse2.0/image_perception_score.csv",
    'eval_path': "/data2/cehou/LLM_safety/eval/test/only_img/",
    'train_type': 'classification',
    'safety_epochs': 200,
    'class_num': 2,
    'CNN_lr': 1*1e-6,
    'weight_on': False
    }

data = pd.read_csv(cfg_paras['placepulse_datapath'])
# data = data[data['Category'] == 'safety'].reset_index(drop=True).iloc[:]
# 计算每个类别的样本数量
# data['label'] = data['Score'] * 100 // (100 / cfg_paras['class_num'])
# label_counts = Counter(data['label'])
# total_samples = len(data)
# class_weights = [0 if label_counts[i] == 0 else total_samples / label_counts[i] for i in range(cfg_paras['class_num'])]
# cfg_paras['class_weights'] = class_weights
run['paras'] = cfg_paras

# data['Score'] = (data['Score'] - data['Score'].min()) / (data['Score'].max() - data['Score'].min())
data_ls = data[data['label'] != 0]
data_ls.loc[data_ls[data_ls['label'] == -1].index, 'label'] = 0
transform = get_transforms(cfg_paras['size'])
split_num = int(len(data_ls) * 0.7)

train_dataset = SafetyPerceptionDataset(data_ls[:split_num], transform=transform, paras=cfg_paras)
valid_dataset = SafetyPerceptionDataset(data_ls[split_num:], transform=transform, paras=cfg_paras)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg_paras['batch_size'], shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg_paras['batch_size'])


train_model(train_loader, valid_loader, cfg_paras)