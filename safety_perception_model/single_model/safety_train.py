# python /code/LLM-crime/safety_perception_model/single_model/safety_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("/code/LLM-crime/single_model")
from torch.utils.data import Dataset
from models import TransformerRegressionModel, ViTClassifier
from PIL import Image
import torchvision.transforms as transforms
from safety_perception_dataset import *
import neptune
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
        input_dim = 28800
        model_dim = 2048
        num_heads = 8  
        num_layers = 6  
        output_dim = 6  

        model = TransformerRegressionModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(paras['device'])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=paras["CNN_lr"])
        
    elif paras['train_type'] == 'classification':
        model = ViTClassifier(num_classes=20,input_dim=360000).to(paras['device'])
        # class_weights = torch.tensor([1.0, 2.0, 3.0])  # 根据类别数量设置权重
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            inputs = inputs.to(paras['device'])
            labels = labels.to(paras['device']).long()
            # print("inputs: ", inputs)
            # print("labels: ", labels)
            # print("image_path: ", image_path)

            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs,labels in valid_loader:
                inputs = inputs.to(paras['device'])
                labels = labels.to(paras['device']).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Record predictions and true labels
                if epoch == 0:
                    all_preds = predicted.cpu().numpy()
                    all_labels = labels.cpu().numpy()
                else:
                    all_preds = np.concatenate((all_preds, predicted.cpu().numpy()))
                    all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

                # Calculate confusion matrix
                # cm = confusion_matrix(labels.cpu(), predicted.cpu())
                # # Plot confusion matrix
                # plt.figure(figsize=(10, 8))
                # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                # plt.xlabel("Predicted")
                # plt.ylabel("True")
                # plt.title(f"Confusion Matrix epoch {epoch+1}")
                # cm_savepath = os.path.join(paras['eval_path'], 'valid_cm')
                # if not os.path.exists(cm_savepath):
                #     os.makedirs(cm_savepath)
                # plt.savefig(os.path.join(cm_savepath, f"confusion_matrix_epoch_{epoch+1}_acc_{correct/total:0.4f}.png"))
                # plt.close()
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_running_loss/train_loader.batch_size:.4f}, Validation Loss: {val_running_loss/valid_loader.batch_size:.4f}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
        if count_after_best > paras['early_stopping_threshold']:
            print("Early Stopping!")
            break

image_size = (300,400)
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
    'device':torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
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
    'early_stopping_threshold':999,
    
    # safety perception
    # 'CLIP_model_path': "/data2/cehou/LLM_safety/LLM_models/clip_model/test/model_baseline_best.pt",
    'variables_save_paths': f"/data2/cehou/LLM_safety/middle_variables/test",
    'safety_model_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/only_img/",
    'placepulse_datapath': "/data2/cehou/LLM_safety/PlacePulse2.0/image_perception.csv",
    'eval_path': "/data2/cehou/LLM_safety/eval/test/only_img/",
    'train_type': 'classification',
    'safety_epochs': 200,
    'CNN_lr': 1e-4,    
    }

data = pd.read_csv("/data2/cehou/LLM_safety/PlacePulse2.0/image_perception.csv")
data = data.iloc[:12000]
data_ls = data[data['Category'] == 'safety']
transform = get_transforms(image_size)
split_num = int(len(data_ls) * 0.8)

train_dataset = SafetyPerceptionDataset(data_ls[:split_num], transform=transform)
valid_dataset = SafetyPerceptionDataset(data_ls[split_num:], transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

train_model(train_loader, valid_loader, cfg_paras)