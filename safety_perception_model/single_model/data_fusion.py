# python /code/LLM-crime/safety_perception_model/single_model/data_fusion.py

import os
# import cv2
import timm
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import numpy as np
import pandas as pd
# import albumentations as A
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import neptune
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from tqdm import tqdm
from safety_perception_dataset import *
from collections import Counter

import sys
sys.path.append("/code/LLM-crime")
from custom_clip_train import CLIPModel, CLIPDataset, build_loaders, make_prediction
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, transforms, cfg_paras):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = dataframe['GSV_path']
        self.captions = list(dataframe['text_description_short'])   
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=cfg_paras['max_length']
        )
        self.labels = dataframe['label']
        self.transforms = transforms
        self.img_type = cfg_paras['img_type']
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        image = self.get_img(idx)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        item['image'] = torch.tensor(image).permute(0, 1, 2).float()
        item['text_description_short'] = self.captions[idx]
        item['label'] = self.labels[idx]
        
        return item

    def get_img(self,idx):
        if self.img_type == 'GSV':
            for i,path in enumerate(self.image_filenames[idx]):
                if i == 0:
                    GSV_img = np.array(Image.open(path))
                else:
                    GSV_img = np.concatenate((GSV_img, np.array(Image.open(path))), axis=1)

            # visualization
            # plt.imshow(GSV_img)
            # plt.title('GSV from original dataset')
            # plt.axis('off')
            return Image.fromarray(GSV_img)
        elif self.img_type == 'PlacePulse':
            GSV_path = self.image_filenames[idx]
            GSV_img = np.array(Image.open(GSV_path))
            return Image.fromarray(GSV_img)

    def __len__(self):
        return len(self.captions)


def get_transforms(cfg_paras, mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(cfg_paras['size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(cfg_paras['size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )    


def build_loaders(dataframe, tokenizer, mode, cfg_paras):
    transforms = get_transforms(mode=mode, cfg_paras=cfg_paras)
    dataset = CLIPDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        transforms=transforms,
        cfg_paras=cfg_paras
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg_paras['batch_size'],
        num_workers=cfg_paras['num_workers'],
        shuffle=True if mode == "train" else False,
    )
    return dataloader

class ProjectionHead(nn.Module):
    def __init__(
        self,
        cfg_paras,
        data_type
    ):
        super().__init__()
        if data_type == 'image':
            self.projection = nn.Linear(cfg_paras['image_embedding'], cfg_paras['projection_dim'])
        elif data_type == 'text':
            self.projection = nn.Linear(cfg_paras['text_embedding'], cfg_paras['projection_dim'])
        # self.projection = nn.Linear(cfg_paras['embedding_dim'], cfg_paras['projection_dim'])
        self.gelu = nn.GELU()
        self.fc = nn.Linear(cfg_paras['projection_dim'], cfg_paras['projection_dim'])
        self.dropout = nn.Dropout(cfg_paras['dropout'])
        self.layer_norm = nn.LayerNorm(cfg_paras['projection_dim'])
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
# class ImageEncoder(nn.Module):
#     """
#     Encode images to a fixed size vector
#     """

#     def __init__(
#         self, 
#         cfg_paras
#     ):
#         super().__init__()
        
#         self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
#         self.model.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(8, 8))  # Adjust for 112x112 input
#         self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
#         self.image_projection = ProjectionHead(cfg_paras, data_type='image')
        
#         for p in self.model.parameters():
#             p.requires_grad = cfg_paras['trainable']

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         # return features.view(features.size(0), -1)
#         print(features.shape)
#         img_embeddings = self.image_projection(features.view(features.size(0), -1))
#         return img_embeddings
    
class ImageEncoder(nn.Module):
    def __init__(self, cfg_paras):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.image_projection = ProjectionHead(cfg_paras, data_type='image')

    def forward(self, x):
        x = self.model(x) # (batch_size, 2048, 1, 1)
        x = torch.flatten(x, 1)  # Flatten the output tensor
        img_embeddings = self.image_projection(x)
        return img_embeddings

    
class TextEncoder(nn.Module):
    def __init__(self, 
                 cfg_paras
                 ):
        super().__init__()
        if cfg_paras['pretrained']:
            self.model = DistilBertModel.from_pretrained(cfg_paras['text_encoder_model'])
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = cfg_paras['trainable']

        self.text_projection = ProjectionHead(cfg_paras, data_type='text')
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # return last_hidden_state[:, self.target_token_idx, :]
        text_embeddings = self.text_projection(last_hidden_state[:, self.target_token_idx, :])
        return text_embeddings
    
def cal_features(train_loader, image_encoder, text_encoder, cfg_paras):
        image_features = []
        text_features = []
        labels = []
        # Set the model to evaluation mode
        image_encoder.eval()
        text_encoder.eval()

        # Disable gradient calculation for inference
        with torch.no_grad():
            for batch in tqdm(train_loader, total=len(train_loader)):
                batch = {k: v.to(cfg_paras['device']) for k, v in batch.items() if k != "text_description_short"}
                img_feature = image_encoder(batch["image"])
                text_feature = text_encoder(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                labels.append(batch["label"].cpu().numpy())
                image_features.append(img_feature.cpu().numpy())
                text_features.append(text_feature.cpu().numpy())

        # Convert the list of features to a numpy array
        image_features = np.concatenate(image_features, axis=0)
        text_features = np.concatenate(text_features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return image_features, text_features, labels


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.key_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.value_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.scale = embed_dim ** 0.5

    def forward(self, query, key, value):
        # query, key, value = query.cuda(), key.cuda(), value.cuda()
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Compute attention weights
        attn_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class Expert(nn.Module):
    """专家网络，用于处理文本或图像特征"""
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim).cuda() 
    
    def forward(self, x):
        return F.relu(self.fc(x))

class MoE(nn.Module):
    def __init__(self, text_dim, image_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        
        # 文本和图像的专家网络
        self.text_expert = Expert(text_dim, output_dim)
        self.image_expert = Expert(image_dim, output_dim)
        
        # 门控网络，用于决定每个专家的权重
        self.gate = nn.Linear(text_dim + image_dim, num_experts).cuda()   # 门控网络的输入是文本和图像的拼接

    def forward(self, text_features, image_features):
        # 将文本和图像特征拼接
        combined_features = torch.cat((text_features, image_features), dim=-1)  # shape: (batch_size, text_dim + image_dim)
        
        # 计算门控网络的权重
        gate_weights = F.softmax(self.gate(combined_features), dim=-1) # shape: (batch_size, num_experts)
        
        # 分别通过文本和图像的专家网络处理输入
        text_output = self.text_expert(text_features)  # shape: (batch_size, output_dim)
        image_output = self.image_expert(image_features)  # shape: (batch_size, output_dim)
        
        # 将专家输出堆叠成一个 tensor
        expert_outputs = torch.stack([text_output, image_output], dim=1)  # shape: (batch_size, num_experts, output_dim)
        
        # 通过门控权重进行加权求和
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # shape: (batch_size, output_dim)
        
        return output

def data_fusion(image_features, text_features, method):
    
    if method == 'concat':
        output = np.concatenate([image_features, text_features], axis=1)
        output = torch.tensor(output)
        
    elif method == 'cross_attention':
        embed_dim = 256
        query = torch.tensor(text_features)  # Batch=16, Sequence Length=10, Embedding=64
        key = torch.tensor(image_features)
        value = torch.tensor(image_features)

        cross_attn = CrossAttention(embed_dim)
        output, attn_weights = cross_attn(query, key, value)
        output = output
    
    elif method == 'MoE':
        text_dim = 256
        image_dim = 256
        output_dim = 512
        num_experts = 2  # 两个专家：一个处理文本，一个处理图像

        # 创建 MoE 模型
        moe = MoE(text_dim, image_dim, output_dim, num_experts)

        # 通过模型进行前向传播
        output = moe(torch.tensor(text_features), torch.tensor(image_features))

        print("Output shape:", output.shape)  # Expected: (32, output_dim)
        
    return output

class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        # self.projection = nn.Linear(cfg_paras['embedding_dim'], cfg_paras['projection_dim'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.dropout(x)        
        x = self.layer_norm(x)
        x = self.fc(x)
        return x
    
class ClassificationDataset(Dataset):
    def __init__(self, fused_feature, labels):
        self.fused_feature = fused_feature
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'fused_feature': self.fused_feature[idx],
            'label': self.labels[idx]
        }

def train_classification_head(classifier_head, train_loader, valid_loader, paras, save=True):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_head.parameters(), lr=paras['lr'])

    # 训练循环
    best_loss = float('inf')
    count_after_best = 0
    train_loss_list = []
    valid_loss_list = []
    accuracy_list = []
    for epoch in range(paras['num_epochs']):
        classifier_head.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, total=len(train_loader)):
            combined_features = batch['fused_feature'].to(paras['device'])
            labels = batch['label'].to(paras['device'])
            
            outputs = classifier_head(combined_features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # valid
        classifier_head.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                combined_features = batch['fused_feature'].to(paras['device'])
                labels = batch['label'].to(paras['device'])
                
                outputs = classifier_head(combined_features)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_loss_list.append(valid_loss)
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print(f"Epoch [{epoch+1}/{paras['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%")
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save:
                torch.save(classifier_head.state_dict(), f"{paras['model_savepath']}/{paras['model_savename']}.pth")
                
        
        count_after_best += 1
        if count_after_best > paras['early_stopping_threshold']:
            # 将数组存储到文件中
            np.save(os.path.join(paras['eval_path'], 'train_loss.npy'), np.array(train_loss_list))
            np.save(os.path.join(paras['eval_path'], 'valid_loss.npy'), np.array(valid_loss_list))
            print(f"Early Stopping!, save loss into {paras['eval_path']}")
            
            # evaluation          
            return classifier_head, accuracy
        
def eval(model, test_loader, paras):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            combined_features = batch['fused_feature'].to(paras['device'])
            labels = batch['label'].to(paras['device'])
            
            outputs = model(combined_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix

    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nF1 Score: {f1:.2f}')
    plt.savefig(f"{paras['eval_path']}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.clf()
    
    # PCA plot
    combined_features = test_loader.dataset.fused_feature
    pca = PCA(n_components=2)
    combined_feature_pca = pca.fit_transform(combined_features)
    # print(combined_features.shape, test_loader.dataset.labels.shape)
    sns.scatterplot(x=combined_feature_pca[:,0], y=combined_feature_pca[:,1], hue=test_loader.dataset.labels)
    plt.title("PCA of Fused Feature")
    plt.savefig(f"{paras['eval_path']}/PCA.png", dpi=300, bbox_inches='tight')
    plt.clf()
    print(f"F1 Score: {f1:.2f}")
    return f1

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
    'image_embedding':2048,
    'text_embedding':768,
    'max_length':512,
    'size':(224,224),
    
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
    'safety_model_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model",
    'placepulse_datapath': "/data2/cehou/LLM_safety/PlacePulse2.0/image_perception_score.csv",
    'eval_path': "/data2/cehou/LLM_safety/eval/test",
    'train_type': 'classification',
    'safety_epochs': 200,
    'batch_size_safety': 256,
    'CNN_lr': 1e-2,
    }
        
    data = pd.read_csv(cfg_paras['dataset_path'])
    data_nonezero = data[data['label'] != 0]
    data_nonezero_idx = data[data['label'] != 0].index
    # img_feature_nonezero = img_feature[data_nonezero_idx,:]
    data_nonezero = data_nonezero.reset_index(drop=True)
    data_nonezero.loc[data_nonezero[data_nonezero['label'] == -1].index,'label'] = 0
    df = data_nonezero[data_nonezero['Category'] == 'safety']
    tokenizer = DistilBertTokenizer.from_pretrained(cfg_paras['text_tokenizer'])

    print("building features")
    feature_extracting_loader = build_loaders(df, tokenizer, mode="train", cfg_paras=cfg_paras)
    # valid_loader = build_loaders(df.iloc[train_num:].reset_index(drop=True), tokenizer, mode="valid", cfg_paras=cfg_paras)
    image_encoder = ImageEncoder(cfg_paras).to(cfg_paras['device'])
    text_encoder = TextEncoder(cfg_paras).to(cfg_paras['device'])
    print("calculating image and text features")
    image_features, text_features, labels = cal_features(feature_extracting_loader, image_encoder, text_encoder, cfg_paras)
    
    # classification model
    # parameters
    paras = {
        
        'fusion_method': 'concat', # data fusion
        
        'batch_size': 256,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), # deep learning
        'lr': 1e-4,
        'num_epochs': 999,
        'repeat_time': 5,
        
        'input_dim': image_features.shape[1] + text_features.shape[1],
        'hidden_dim': 1024,
        'output_dim': 2,
        
        'model_savepath': '/data_nas/cehou/LLM_safety/LLM_models/classifier_head',
        'model_savename': 'classifier_head',
        'eval_path': '/data2/cehou/LLM_safety/classifier/eval',
        
        'early_stopping_threshold':5
    }
    
    # data fusion
    fused_feature = data_fusion(image_features, text_features, method=paras['fusion_method'])
    
    train_num = int(len(fused_feature) * 0.6)
    valid_num = int(len(fused_feature) * 0.2)
    train_dataset = ClassificationDataset(fused_feature, labels[:train_num])
    valid_dataset = ClassificationDataset(fused_feature[train_num:train_num+valid_num], labels[train_num:train_num+valid_num])
    test_dataset = ClassificationDataset(fused_feature[train_num+valid_num:], labels[train_num+valid_num:])

    train_loader = DataLoader(train_dataset, batch_size=paras['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=paras['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=paras['batch_size'])
    
    tuning_paras = 'lr'
    tuning_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for i in tuning_values:
        for j in range(paras['repeat_time']):
            paras['model_savename'] = f'classifier_head_{tuning_paras}_{i}_{j}'
            paras['eval_path'] = f"/data2/cehou/LLM_safety/classifier/eval/{tuning_paras}_{i}_{j}"
            if not os.path.exists(paras['eval_path']):
                os.makedirs(paras['eval_path'])
            
            classifier_head = ClassifierHead(paras['input_dim'], paras['hidden_dim'], paras['output_dim']).to(paras['device'])
            classifier_head, accuracy = train_classification_head(classifier_head, train_loader, valid_loader, paras, save=True)
            paras['accuracy'] = accuracy
            
            # evaluation
            f1 = eval(classifier_head, test_loader, paras)
            paras['f1'] = f1
            paras['device'] = 'cuda'
            paras_df = pd.DataFrame(paras, index=[0])
            paras_df.to_csv(f"{paras['eval_path']}/paras.csv", index=False)

        
if __name__ == '__main__':
    main()