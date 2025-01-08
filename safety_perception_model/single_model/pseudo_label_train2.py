# python /code/LLM-crime/safety_perception_model/single_model/pseudo_label_train2.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import sys
sys.path.append("/code/LLM-crime/single_model")
from my_models import TransformerRegressionModel, ResNet50Model, ViTClassifier
from LLM_feature_extractor import LLaVaFeatureExtractor
import transformers
from scipy.stats import norm
from data_fusion import *
from PIL import Image
import torchvision.transforms as transforms
from safety_perception_dataset import *
import neptune
from sklearn.metrics import confusion_matrix, f1_score
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import r2_score
import shutil
from itertools import product
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# 创建模型实例

class LLMImageFeaturePrextractor(nn.Module):
    def __init__(self, process='mean'):
        super(LLMImageFeaturePrextractor, self).__init__()
        self.llava_extractor = LLaVaFeatureExtractor()
        self.conv_dim1 = nn.Conv2d(3, 1, kernel_size=1)  # 输入3通道，输出1通道
        self.conv_dim2 = nn.Conv2d(3, 3, kernel_size=1)  # 输入3通道，输出3通道
        self.process = process
    
    def forward(self, x):
        img_feature = self.llava_extractor.image_extractor(x)
        
        if self.process == 'mean_dim1':
            img_feature = img_feature.mean(dim=(1))
        if self.process == 'mean_dim2':
            img_feature = img_feature.mean(dim=(2))
        if self.process == 'mean':
            img_feature = img_feature.mean(dim=(1,2))
        if self.process == 'max_dim1':
            img_feature = img_feature.max(dim=(1))[0]
        if self.process == 'max_dim2':
            img_feature = img_feature.max(dim=(2))[0]
        if self.process == 'reshape':
            img_feature = img_feature.reshape(-1, img_feature.shape[3], img_feature.shape[4])
        if self.process == 'conv_dim1':
            img_feature = self.conv_dim1(img_feature)
        if self.process == 'conv_dim2':
            img_feature = self.conv_dim2(img_feature)
        return img_feature

class LLMTextFeaturePrextractor(nn.Module):
    def __init__(self):
        super(LLMTextFeaturePrextractor, self).__init__()
        self.llava_extractor = LLaVaFeatureExtractor()
    
    def forward(self, x):
        text_feature = self.llava_extractor.text_extractor(x)
        return text_feature

def get_transforms(resize_size):
    return transforms.Compose(
        [
            transforms.Resize((resize_size[0], resize_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  

class ImageExtractor(nn.Module):
    def __init__(self, pretrained_model='resnet18'):
        super(ImageExtractor, self).__init__()
        self.pretrained_model = pretrained_model
        if pretrained_model == 'ViT':
            # self.model = models.vit_b_16(pretrained=True)
            # self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.model = create_feature_extractor(models.vit_b_16(pretrained=True), return_nodes={'getitem_5': 'features'})

        if pretrained_model == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # 去掉最后的全连接层
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        if pretrained_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # 去掉最后的全连接层
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
    def forward(self, x):
        # 输入图像 x，返回提取的特征
        with torch.no_grad():  # 禁用梯度计算
            if self.pretrained_model == 'ViT':
                features = self.model(x)['features']
            else:
                features = self.model(x)
                features = features.view(features.size(0), -1)
            # if features.dim() == 4:
            #     features = F.adaptive_avg_pool2d(features, (1, 1))
        # 返回特征的展平（flatten）形式
        return features

class TextExtractor(nn.Module):
    def __init__(self, pretrained_model='Bert'):
        super(TextExtractor, self).__init__()
        if pretrained_model == 'Bert':
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        if pretrained_model == 'GPT2':
            self.model = transformers.GPT2Model.from_pretrained('gpt2')
        if pretrained_model == 'DistilBert':
            self.model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        if pretrained_model == 'LLM':
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')

        self.target_token_idx = 0
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            
            last_hidden_state = last_hidden_state.mean(dim=1)
            # print("last_hidden_state: ", last_hidden_state.shape)
        # return last_hidden_state[:, self.target_token_idx, :]
        return last_hidden_state
    
    
class Adaptor(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim,
        data_type
    ):
        super(Adaptor, self).__init__()
        if data_type == 'image':
            self.projection = nn.Linear(input_dim, projection_dim)
        elif data_type == 'text':
            self.projection = nn.Linear(input_dim, projection_dim)
        # self.projection = nn.Linear(cfg_paras['embedding_dim'], cfg_paras['projection_dim'])
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Mixer(nn.Module):
    def __init__(self, output_dim, process='concat'):
        super(Mixer, self).__init__()
        self.process = process
        self.output_dim = output_dim

    def forward(self, image_features, text_features):
        # print(self.process)
        if self.process == 'concat':
            output = torch.cat((image_features, text_features), dim=1)
        
        elif self.process == 'cross_attention':
            query = torch.tensor(text_features)  # Batch=16, Sequence Length=10, Embedding=64
            key = torch.tensor(image_features)
            value = torch.tensor(image_features)

            cross_attn = CrossAttention(self.output_dim)
            output, attn_weights = cross_attn(query, key, value)
    
        elif self.process == 'MoE':
            num_experts = 2 
            moe = MoE(image_features.shape[1], text_features.shape[1], self.output_dim, num_experts)
            output = moe(text_features, image_features)

        elif self.process == 'fc':
            output = torch.cat((image_features, text_features), dim=1)
            output = nn.Linear(output.shape[1], self.output_dim).cuda()(output)
            
        return output

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 一个简单的全连接层作为分类器
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 输入适配后的特征向量，输出分类结果
        return self.fc(x)

class MultiModalModel(nn.Module):
    def __init__(self, image_extractor, text_extractor, image_adaptor, text_adaptor, mixer, classifier):
        super(MultiModalModel, self).__init__()
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.image_adaptor = image_adaptor
        self.text_adaptor = text_adaptor

        self.mixer = mixer
        self.classifier = classifier

    def forward(self, x_img, x_text, attention_mask):
        # 先通过extractor提取特征，再通过adaptor处理，最后分类
        img_features = self.image_extractor(x_img)
        text_features = self.text_extractor(x_text, attention_mask)
        adapted_img_features = self.image_adaptor(img_features)
        adapted_text_features = self.text_adaptor(text_features)
        mixed_features = self.mixer(adapted_img_features, adapted_text_features)
        # print("extracted feature: ", features.shape)
        # print("adapted feature: ", adapted_features.shape)
        output = self.classifier(mixed_features)
        # print("final feature", output.shape)
        return output
    
# Early Stopping类
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
    
def train(model, pbar, criterion, optimizer, LLM_model=None):
    if LLM_model is not None:
        LLM_pre_extractor = LLM_model
    model.train()  # 切换到训练模式
    running_loss = 0.0
    for batch_idx, (data, (description, attention_mask), target) in enumerate(pbar):
        if LLM_model is not None:
            data = LLM_pre_extractor([data[i] for i in range(len(data))])
    
        # 将数据和目标移到GPU（如果有的话）
        data, description, attention_mask, target = data.cuda(), description.cuda().long(), attention_mask.cuda().long(), target.cuda()
        optimizer.zero_grad()  # 清零梯度
        output = model(data, description, attention_mask)  # 获取模型输出
        loss = criterion(output, target)

        # target_one_hot = F.one_hot(target, num_classes=2).float()
        # loss = criterion(output, target_one_hot)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss


def eval(model, valid_loader, criterion, LLM_model=None):
    if LLM_model is not None:
        LLM_pre_extractor = LLM_model
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for data, (description, attention_mask), target in valid_loader:
            if LLM_model is not None:
                data = LLM_pre_extractor([data[i] for i in range(len(data))])
            data, description, attention_mask, target = data.cuda(), description.cuda().long(), attention_mask.cuda().long(), target.cuda()
            output = model(data, description, attention_mask)  # 获取模型输出
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss 


def pseudo_label_generation(model, test_loader, criterion, confidence_threshold=0.75):
    print("Pseudo label generation")
    with torch.no_grad():
        all_pseudo_labels = []
        all_probabilities = []
        for data, (description, attention_mask), target in tqdm(test_loader):
            data, description, attention_mask, target = data.cuda(), description.cuda().long(), attention_mask.cuda().long(), target.cuda()
            output = model(data, description, attention_mask)  # 获取模型输出
            loss = criterion(output, target)

            probabilities = torch.softmax(output, dim=1)
            pseudo_labels = torch.argmax(probabilities, dim=1)
            all_pseudo_labels.extend(pseudo_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_pseudo_labels = torch.tensor(all_pseudo_labels)
    probabilities = torch.tensor(all_probabilities)
    print(probabilities)
    confidence_threshold = confidence_threshold
    max_probs, _ = torch.max(probabilities, dim=1)
    high_confidence_mask = max_probs > confidence_threshold
    # 从all_pseudo_labels中挑选等量的分类样本作为mask
    class_counts = Counter(all_pseudo_labels.numpy())
    min_class_count = min(class_counts.values())

    balanced_mask = torch.zeros_like(high_confidence_mask, dtype=torch.bool)
    for label in class_counts.keys():
        label_indices = torch.where(all_pseudo_labels == label)[0]
        selected_indices = label_indices[torch.randperm(len(label_indices))[:min_class_count]]
        balanced_mask[selected_indices] = True

    balanced_high_confidence_mask = high_confidence_mask & balanced_mask
    
    # y_high_confidence_idx = all_pseudo_labels[high_confidence_mask]
    y_high_confidence_idx = torch.where(balanced_high_confidence_mask == 1)[0]
    print(balanced_high_confidence_mask, y_high_confidence_idx)
    return all_pseudo_labels, y_high_confidence_idx

def make_loaders(data_ls, data_test, parameters):

    transform = get_transforms((224,224))
    train_num = int(len(data_ls) * 0.6)
    valid_num = int(len(data_ls) * 0.2)

    if parameters['LLM_loaded'] == True:
        LLM_pre_extractor = LLMImageFeaturePrextractor(process=parameters['LLM_image_feature_process'])
        train_dataset = MultimodalSafetyPerceptionDataset(data_ls[:train_num], tokenizer=parameters['text_feature_extractor'], paras=parameters, SVI_type=parameters['SVI_type'])
        valid_dataset = MultimodalSafetyPerceptionDataset(data_ls[train_num:train_num+valid_num], tokenizer=parameters['text_feature_extractor'], paras=parameters, SVI_type=parameters['SVI_type'])
        test_dataset = MultimodalSafetyPerceptionDataset(data_test, tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters, SVI_type=parameters['SVI_type'])
    else:
        LLM_pre_extractor = None
        train_dataset = MultimodalSafetyPerceptionDataset(data_ls[:train_num], tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters, SVI_type=parameters['SVI_type'])
        valid_dataset = MultimodalSafetyPerceptionDataset(data_ls[train_num:train_num+valid_num], tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters, SVI_type=parameters['SVI_type'])
        test_dataset = MultimodalSafetyPerceptionDataset(data_test, tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters, SVI_type=parameters['SVI_type'])
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=parameters['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batch_size'])
    return train_loader, valid_loader, test_loader, LLM_pre_extractor

def model_test(model, test_loader, LLM_model=None):
    if LLM_model is not None:
        LLM_pre_extractor = LLM_model
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算，节省内存
        for data, (description, attention_mask), target in test_loader:
            if LLM_model is not None:
                data = LLM_pre_extractor([data[i] for i in range(len(data))])
            data, description, attention_mask, target = data.cuda(), description.cuda().long(), attention_mask.cuda().long(), target.cuda().long()
            output = model(data, description, attention_mask)  # 获取模型输出
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 计算F1分数
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score: {f1:.4f}")
    return f1, cm, all_preds, all_labels

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    return None

def main(variables_dict=None):
    # parameters
    parameters = {
        'train_type': "classification",
        'placepulse_datapath': "/data2/cehou/LLM_safety/img_text_data/finished/dataset_60_female_HongKong_traffic accident_all_4685.pkl",
        'safety_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/multimodal/",
        'safety_model_save_name':"model_baseline.pt",
        'subfolder_name': 'baseline',
        'load_pretrain_model': "/data2/cehou/LLM_safety/LLM_models/safety_perception_model/multimodal/diff_concat_20241230/lr_1e-05_adaptor_output_dim_256_mix_process_concat/model_baseline_lr_1e-05_adaptor_output_dim_256_mix_process_concat.pt",
        'SVI_type': 'GSV',
        
        # model training parameters
        'num_epochs': 199,
        'visual_feature_extractor': 'ViT',
        'text_feature_extractor': 'Bert',
        'batch_size': 128,
        'image_input_dim': 768,
        'text_input_dim': 768,
        'adaptor_output_dim': 256,
        'mix_process': 'concat',
        'mixer_output_dim': 512,
        'num_classes': 2,
        'lr': 1e-6,
        'LLM_loaded': False,
        'LLM_image_feature_process': 'mean_dim1',
        'train_loss_list': [],
        'val_loss_list': [],
        'accuracy': None,
        'f1_score': None
    }
    
    if variables_dict is not None:
        for key, value in variables_dict.items():
            parameters[key] = value
            
        parameters['safety_model_save_name'] = f"model_baseline_{parameters['subfolder_name']}.pt"
        
    if not os.path.exists(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'])):
        os.makedirs(os.path.join(parameters['safety_save_path'], parameters['subfolder_name']))
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_extractor = ImageExtractor(pretrained_model=parameters['visual_feature_extractor']) # [128, 512]
    text_extractor = TextExtractor(pretrained_model=parameters['text_feature_extractor']) # [128, 768]
    image_adaptor = Adaptor(input_dim=parameters['image_input_dim'], projection_dim=parameters['adaptor_output_dim'], data_type='image') # [128, 256]
    text_adaptor = Adaptor(input_dim=parameters['text_input_dim'], projection_dim=parameters['adaptor_output_dim'], data_type='text') # [128, 256]
    mixer = Mixer(output_dim=parameters['mixer_output_dim'], process=parameters['mix_process']) # [128, 512]
    classifier = Classifier(input_dim=parameters['mixer_output_dim'], num_classes=parameters['num_classes']) # [128, 2]
    model = MultiModalModel(image_extractor, text_extractor, image_adaptor, text_adaptor, mixer, classifier).to(device)
    if parameters['load_pretrain_model'] is not None:
        print("Load pretrain model")
        model.load_state_dict(torch.load(parameters['load_pretrain_model']), strict=False)
    print(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])
                        
    data_fulfilled = False
    confidence_list = np.zeros(50)
    confidence_list[:10] = 0.9
    confidence_list[10:20] = 0.8
    confidence_list[20:] = 0.85
    
    for i, confidence_threshold in enumerate(confidence_list):
        parameters['train_loss_list'] = []
        parameters['val_loss_list'] = []
        os.makedirs(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}"), exist_ok=True)
        if i == 0:
            if parameters['load_pretrain_model'] is None:
                print('train the model')
                data = pd.read_csv(parameters['placepulse_datapath'])
                mu, std = norm.fit(data['Score'])
                std_threshold = 2
                data['label'] = 0
                data = data[(data['Score'] < (mu - std)) | (data['Score'] > (mu + std))]
                
                data.loc[data[data['Score'] > mu + std_threshold * std].index, 'label'] = 1
                data.loc[data[data['Score'] < mu - std_threshold * std].index, 'label'] = -1
                data_ls = data[data['label'] != 0].reset_index(drop=True)
                data_test = data[data['label'] == 0].reset_index(drop=True)
                data_ls['label'] = data_ls['label'].apply(lambda x: 1 if x == 1 else 0)    
                
                train_loader, valid_loader, test_loader, LLM_pre_extractor = make_loaders(data_ls, data_test, parameters)
 
                
            else:
                if parameters['placepulse_datapath'].endswith('.csv'):
                    data = pd.read_csv(parameters['placepulse_datapath'])
                elif parameters['placepulse_datapath'].endswith('.pkl'):
                    data = pd.read_pickle(parameters['placepulse_datapath'])
                    
                data['label'] = 0
                data = data.rename(columns={'text_description_age': 'text_description_short',
                                            'panoid': 'Image_ID',})
                data_ls = data.sample(n=parameters['batch_size'], random_state=1).reset_index(drop=True)
                data_test = data.copy()
                train_loader, valid_loader, test_loader, LLM_pre_extractor = make_loaders(data_ls, data_test, parameters, )
                pseudo_labels, y_high_confidence_idx = pseudo_label_generation(model, test_loader, criterion, confidence_threshold=0.8)
                pseudo_labels = pseudo_labels.cpu().numpy()
                y_high_confidence_idx = y_high_confidence_idx.cpu().numpy()
                print("update data labels")
                
                # print(pseudo_labels, y_high_confidence_idx)
                for j,idx in tqdm(enumerate(y_high_confidence_idx)):
                    data_test.loc[idx, 'label'] = pseudo_labels[j]
                data_update = data_test.iloc[y_high_confidence_idx].reset_index(drop=True)
                data_ls = data_update.copy()
                data_test = data_test[~data_test.index.isin(y_high_confidence_idx)].reset_index(drop=True)
                print(f"Confidence threshold: {0.85}, High confidence samples: {len(y_high_confidence_idx)}, updated data samples: {len(data_ls)}, rest test samples: {len(data_test)}")


        print(f"Data size: {len(data_ls)}, Test data size: {len(data_test)}")

        if i > 0:
            # initialize
            train_loader, valid_loader, test_loader, LLM_pre_extractor = make_loaders(data_ls, data_test, parameters)
        
        early_stopping = EarlyStopping(patience=5, verbose=True)
        for epoch in range(parameters["num_epochs"]):    
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{parameters['num_epochs']}",unit="batch", mininterval=2.0)
            running_loss = train(model, pbar, criterion, optimizer, LLM_model=LLM_pre_extractor)
            val_loss = eval(model, valid_loader, criterion, LLM_model=LLM_pre_extractor)

            # 输出当前epoch的训练和验证损失
            print(f"Epoch [{epoch + 1}/{parameters['num_epochs']}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss / len(valid_loader):.4f}")
            parameters['train_loss_list'].append(running_loss / len(train_loader))
            parameters['val_loss_list'].append(val_loss / len(valid_loader))
            # 触发早停机制
            early_stopping(val_loss / len(valid_loader))
            if early_stopping.best_loss == val_loss / len(valid_loader):
                torch.save(model.state_dict(), os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  parameters['safety_model_save_name']))
                print("Model saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  parameters['safety_model_save_name']))

            if early_stopping.early_stop:
                print("Early stopping")
                break        

        torch.save(model.state_dict(), os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  parameters['safety_model_save_name']))
        print("Model saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  parameters['safety_model_save_name']))
        
        f1_train, cm_train, _, _ = model_test(model, train_loader, LLM_model=LLM_pre_extractor)
        fig, ax = plt.subplots()
        sns.heatmap(cm_train, annot=True, ax=ax, cmap='Blues', fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix in train set at round {i}')
        plt.savefig(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  'confusion_matrix_train.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        
        f1, cm, all_preds, all_labels = model_test(model, test_loader, LLM_model=LLM_pre_extractor)
        print(cm)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix in text set at round {i}')
        plt.savefig(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  'confusion_matrix_test.png'), dpi=300, bbox_inches='tight')

        parameters['accuracy'] = cm.diagonal().sum() / cm.sum()
        parameters['f1_score_train'] = f1_train
        parameters['f1_score_test'] = f1

        pd.DataFrame(parameters).to_csv(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  'parameters.csv'))
        print("Parameters saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], f"round_{i}",  'parameters.csv'))    


        if data_fulfilled:
            print("Data fulfilled")
            break

        pseudo_labels, y_high_confidence_idx = pseudo_label_generation(model, test_loader, criterion, confidence_threshold=0.95)
        pseudo_labels = pseudo_labels.cpu().numpy()
        y_high_confidence_idx = y_high_confidence_idx.cpu().numpy()
        print("update data labels")
        
        # print(pseudo_labels, y_high_confidence_idx)
        for i,idx in tqdm(enumerate(y_high_confidence_idx)):
            data_test.loc[idx, 'label'] = pseudo_labels[i]
        data_update = data_test.iloc[y_high_confidence_idx].reset_index(drop=True)
        data_ls = pd.concat([data_ls, data_update], axis=0).reset_index(drop=True)
        # data_ls['label'] = data_ls['label'].apply(lambda x: 1 if x == 1 else 0)
        data_test = data_test[~data_test.index.isin(y_high_confidence_idx)].reset_index(drop=True)
        print(f"Confidence threshold: {confidence_threshold}, High confidence samples: {len(y_high_confidence_idx)}, updated data samples: {len(data_ls)}, rest test samples: {len(data_test)}")
        if len(data_ls) > 0.2 * (len(data_ls) + len(data_test)):
            data_fulfilled = True


if __name__ == '__main__':
    variables_dict = { 'lr':[1e-5], # np.linspace(1e-6, 1e-5, 5), # [0.001, 1e-4, 1e-5, 1e-6, 1e-7]
                      'adaptor_output_dim':[256],
                    #   'mix_process':[ "MoE", "cross_attention", 'concat', 'fc'],
                    'mix_process':['concat'],
                    #   'mixer_output_dim':[512]
                      }
    combinations = list(product(*variables_dict.values()))

    for combination in tqdm(combinations):
        print(combination)
        input_dict = dict(zip(variables_dict.keys(), combination))
        input_dict['subfolder_name'] = '_'.join([f"{key}_{value}" for key, value in input_dict.items()])
        input_dict['safety_save_path'] = f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/multimodal/pseudo_label_from_pretrain_20250108"
        os.makedirs(input_dict['safety_save_path'], exist_ok=True)

        input_dict['mixer_output_dim'] = input_dict['adaptor_output_dim'] * 2

        # 根据模型的不同改变input_dim
        if input_dict['mix_process'] == 'cross_attention':
            input_dict['mixer_output_dim'] = input_dict['adaptor_output_dim']
            
        main(input_dict)