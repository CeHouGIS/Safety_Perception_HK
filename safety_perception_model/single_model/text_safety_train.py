# python /code/LLM-crime/safety_perception_model/single_model/text_safety_train.py
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
from data_fusion import *
from PIL import Image
import torchvision.transforms as transforms
from safety_perception_dataset import *
import neptune
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import r2_score
import shutil
from itertools import product
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 创建模型实例

class LLMTextFeaturePrextractor(nn.Module):
    def __init__(self):
        super(LLMTextFeaturePrextractor, self).__init__()
        self.llava_extractor = LLaVaFeatureExtractor()
    
    def forward(self, x):
        text_feature = self.llava_extractor.text_extractor(x)
        return text_feature

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
        return last_hidden_state[:, self.target_token_idx, :]
    


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

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        # 一个简单的全连接层作为分类器
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 输入适配后的特征向量，输出分类结果
        return self.fc(x)

class TextModel(nn.Module):
    def __init__(self, text_extractor, text_adaptor, classifier):
        super(TextModel, self).__init__()
        self.text_extractor = text_extractor
        self.text_adaptor = text_adaptor
        self.classifier = classifier

    def forward(self, x_text, attention_mask):
        # 先通过extractor提取特征，再通过adaptor处理，最后分类
        text_features = self.text_extractor(x_text, attention_mask)
        adapted_text_features = self.text_adaptor(text_features)
        # print("extracted feature: ", features.shape)
        # print("adapted feature: ", adapted_features.shape)
        output = self.classifier(adapted_text_features)
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
    for batch_idx, ((data, attention_mask), target) in enumerate(pbar):
        if LLM_model is not None:
            data = LLM_pre_extractor([data[i] for i in range(len(data))])
    
        # 将数据和目标移到GPU（如果有的话）
        data, attention_mask, target = data.cuda().long(), attention_mask.cuda().long(), target.cuda()
        optimizer.zero_grad()  # 清零梯度
        output = model(data,attention_mask)  # 获取模型输出
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
        for ((data, attention_mask), target) in valid_loader:
            if LLM_model is not None:
                data = LLM_pre_extractor([data[i] for i in range(len(data))])
            data, attention_mask, target = data.cuda().long(), attention_mask.cuda().long(), target.cuda().long()
            output = model(data,attention_mask)
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss 

def model_test(model, test_loader, LLM_model=None):
    if LLM_model is not None:
        LLM_pre_extractor = LLM_model
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算，节省内存
        for ((data, attention_mask), target) in test_loader:
            if LLM_model is not None:
                data = LLM_pre_extractor([data[i] for i in range(len(data))])
            data, attention_mask, target = data.cuda().long(), attention_mask.cuda().long(), target.cuda().long()
            output = model(data,attention_mask)
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
        'placepulse_datapath': "/data2/cehou/LLM_safety/img_text_data/baseline/tidyed/dataset_baseline_baseline_baseline_baseline_14294_withlabel.csv",
        'safety_save_path' : f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/multimodal/",
        'safety_model_save_name':"model_baseline.pt",
        'subfolder_name': 'baseline',
        
        # model training parameters
        'num_epochs': 400,
        'text_feature_extractor': 'Bert',
        'batch_size': 128,
        'text_input_dim': 768,
        'adaptor_output_dim': 256,
        'num_classes': 2,
        'lr': 0.001,
        'LLM_loaded': True,
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
        
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(parameters['placepulse_datapath'])
    data_ls = data[data['label'] != 0]
    data_ls.loc[data_ls[data_ls['label'] == -1].index, 'label'] = 0
    transform = get_transforms((224,224))
    train_num = int(len(data_ls) * 0.6)
    valid_num = int(len(data_ls) * 0.2)

    if parameters['LLM_loaded'] == True:
        train_dataset = TextSafetyPerceptionDataset(data_ls[:train_num], tokenizer=parameters['text_feature_extractor'], paras=parameters)
        valid_dataset = TextSafetyPerceptionDataset(data_ls[train_num:train_num+valid_num], tokenizer=parameters['text_feature_extractor'], paras=parameters)
        test_dataset = TextSafetyPerceptionDataset(data_ls[train_num+valid_num:], tokenizer=parameters['text_feature_extractor'], paras=parameters)
    else:
        LLM_pre_extractor = None
        train_dataset = TextSafetyPerceptionDataset(data_ls[:train_num], tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters)
        valid_dataset = TextSafetyPerceptionDataset(data_ls[train_num:train_num+valid_num], tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters)
        test_dataset = TextSafetyPerceptionDataset(data_ls[train_num+valid_num:], tokenizer=parameters['text_feature_extractor'], transform=transform, paras=parameters)
        
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=parameters['batch_size'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=parameters['batch_size'])

    text_extractor = TextExtractor(pretrained_model=parameters['text_feature_extractor']) # [128, 768]
    text_adaptor = Adaptor(input_dim=parameters['text_input_dim'], projection_dim=parameters['adaptor_output_dim'], data_type='text') # [128, 256]
    classifier = Classifier(input_dim=parameters['adaptor_output_dim'], num_classes=parameters['num_classes']) # [128, 2]
    model = TextModel(text_extractor, text_adaptor, classifier).to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])
                        
                    
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
            torch.save(model.state_dict(), os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], parameters['safety_model_save_name']))
            print("Model saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], parameters['safety_model_save_name']))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], parameters['safety_model_save_name']))
    print("Model saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], parameters['safety_model_save_name']))
    
    f1, cm, all_preds, all_labels = model_test(model, test_loader, LLM_model=LLM_pre_extractor)
    parameters['accuracy'] = cm.diagonal().sum() / cm.sum()
    parameters['f1_score'] = f1
    
    pd.DataFrame(parameters).to_csv(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], 'parameters.csv'))
    print("Parameters saved at ", os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], 'parameters.csv'))    

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='g')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(os.path.join(parameters['safety_save_path'], parameters['subfolder_name'], 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
   
if __name__ == '__main__':
    variables_dict = {'lr': np.linspace(1e-6, 1e-5, 5),
                      'text_feature_extractor': ['Bert','DistilBert'],
                      'LLM_loaded': [False],}
    combinations = list(product(*variables_dict.values()))

    for combination in tqdm(combinations):
        input_dict = dict(zip(variables_dict.keys(), combination))
        input_dict['subfolder_name'] = '_'.join([f"{key}_{value}" for key, value in input_dict.items()])
        input_dict['safety_save_path'] = f"/data2/cehou/LLM_safety/LLM_models/safety_perception_model/text/only_text_20241224"
        os.makedirs(input_dict['safety_save_path'], exist_ok=True)

        # 根据模型的不同改变input_dim
        if combination[1] == 'resnet50':
            input_dict['image_input_dim'] = 2048
        if combination[1] == 'resnet18':
            input_dict['image_input_dim'] = 512
        if combination[1] == 'ViT':
            input_dict['image_input_dim'] = 768
            
        main(input_dict)