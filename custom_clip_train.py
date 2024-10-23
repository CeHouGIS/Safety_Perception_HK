# python /code/LLM-crime/custom_clip_train.py


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
from tqdm.autonotebook import tqdm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import neptune
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GSV auto download script')
parser.add_argument('--save-model-path', default=f"/data1/cehou_data/LLM_safety/LLM_model/clip_model", type=str,
                    help='batch size')

parser.add_argument('--batch-size', default=20, type=int,
                    help='batch size')
parser.add_argument('--head-lr', default=1e-3, type=float,
                    help='batch size')
parser.add_argument('--image-encoder-lr', default=1e-4, type=float,
                    help='batch size')
parser.add_argument('--text-encoder-lr', default=1e-5, type=float,
                    help='batch size')
parser.add_argument('--weight-decay', default=1e-3, type=float,
                    help='batch size')
parser.add_argument('--patience', default=1, type=float,
                    help='batch size')
parser.add_argument('--factor', default=0.8, type=float,
                    help='batch size')
parser.add_argument('--epochs', default=200, type=int,
                    help='batch size')
parser.add_argument('--image-embedding', default=2048, type=int,
                    help='batch size')
parser.add_argument('--text-embedding', default=768, type=int,
                    help='batch size')
parser.add_argument('--max-length', default=512, type=int,
                    help='batch size')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='batch size')
parser.add_argument('--projection-dim', default=256, type=int,
                    help='batch size')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='batch size')



## Parameters
class Configurations:
    def __init__(self,cfg_paras):
        # args = parser.parse_args()
        # print(args)
        self.debug = cfg_paras['debug'] # False
        self.dataset_path = "/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_501.pkl"
        # dataset_path = "/data_nas/cehou/LLM_safety/dataset_baseline_746.pkl"
        # image_path = "../input/flickr-image-dataset/flickr30k_images/flickr30k_images"
        self.dataset_config = self.dataset_path.split("/")[-1].split("_")
        # save_model_path = f"/data_nas/cehou/LLM_safety/model/model_baseline.pt"
        self.save_model_path = cfg_paras['save_model_path']
        self.save_model_name = f"model_{self.dataset_config[1]}_{self.dataset_config[2]}_{self.dataset_config[3]}_{self.dataset_config[4]}.pt"
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.captions_path = "."
        self.batch_size = cfg_paras['batch_size']
        self.num_workers = 4
        self.head_lr = cfg_paras['head_lr']
        self.image_encoder_lr = cfg_paras['image_encoder_lr']
        self.text_encoder_lr = cfg_paras['text_encoder_lr']
        self.weight_decay = cfg_paras['weight_decay']
        self.patience = cfg_paras['patience']
        self.factor = cfg_paras['factor']
        self.epochs = cfg_paras['epochs']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.early_stopping_threshold = cfg_paras['early_stopping_threshold']

        self.model_name = cfg_paras['model_name']
        self.image_embedding = cfg_paras['image_embedding']
        self.text_encoder_model = cfg_paras['text_encoder_model']
        self.text_embedding = cfg_paras['text_embedding']
        self.text_tokenizer = cfg_paras['text_tokenizer']
        self.max_length = cfg_paras['max_length']

        self.pretrained = True # for both image encoder and text encoder
        self.trainable = True # for both image encoder and text encoder
        self.temperature = cfg_paras['max_length']

        # image size
        # self.size = (int(300), int(400))  # (height, width)
        self.size = cfg_paras['size']

        # for projection head; used for both image and text encoders
        self.projection_dim = cfg_paras['projection_dim']
        self.dropout = cfg_paras['dropout']
        
    
    
## Utils
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms, cfg_paras):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=cfg_paras['max_length']
        )
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
        item['caption'] = self.captions[idx]

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

# call it
# tokenizer = DistilBertTokenizer.from_pretrained(Configurations.text_tokenizer)
# clip_dataset = CLIPDataset(
#     [i['GSV_path'] for i in dataset], 
#     [i['text_description'] for i in dataset], 
#     tokenizer=tokenizer, 
#     transforms=None
# )

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

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, 
        cfg_paras
    ):
        super().__init__()
        self.model = timm.create_model(
            cfg_paras['model_name'], cfg_paras['pretrained'], num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = cfg_paras['trainable']

    def forward(self, x):
        return self.model(x)
    
    
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

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
    
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
    
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        cfg_paras
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg_paras)
        self.text_encoder = TextEncoder(cfg_paras)
        self.image_projection = ProjectionHead(cfg_paras, data_type='image')
        self.text_projection = ProjectionHead(cfg_paras, data_type='text')
        self.temperature = cfg_paras['temperature']

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  0.75*images_loss + 0.25*texts_loss # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
def make_train_valid_dfs(cfg_paras):
    dataframe = pd.read_csv(f"{cfg_paras['captions_path']}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not cfg_paras['debug'] else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode, cfg_paras):
    transforms = get_transforms(mode=mode, cfg_paras=cfg_paras)
    dataset = CLIPDataset(
        [i['GSV_path'] for i in dataframe], 
        [i['text_description_short'] for i in dataframe], 
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


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, cfg_paras):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(cfg_paras['device']) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader, cfg_paras):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(cfg_paras['device']) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def make_prediction(model, test_loader, cfg_paras):
    model.eval()
    predictions = []
    image_embeddings_list = []
    text_embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(cfg_paras['device']) for k, v in batch.items() if k != "caption"}
            image_features = model.image_encoder(batch["image"])
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)
            
            logits = (text_embeddings @ image_embeddings.T) / model.temperature
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            
            image_embeddings_list.extend(image_embeddings.cpu().numpy())
            text_embeddings_list.extend(text_embeddings.cpu().numpy())
    # return predictions
    return image_embeddings_list, text_embeddings_list

# define train function, similar to main()
def clip_train(cfg_paras):
    # CFG = Configurations(cfg_paras)
    df = pd.read_pickle(cfg_paras['dataset_path'])
    
    # train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(cfg_paras['text_tokenizer'])
    
    train_num = int(len(df) * 0.7)
    train_loader = build_loaders(df[:train_num], tokenizer, mode="train", cfg_paras=cfg_paras)
    valid_loader = build_loaders(df[train_num:], tokenizer, mode="valid", cfg_paras=cfg_paras)

    print("use device: ", cfg_paras['device'])
    model = CLIPModel(cfg_paras).to(cfg_paras['device'])
    params = [
        {"params": model.image_encoder.parameters(), "lr": cfg_paras['image_encoder_lr']},
        {"params": model.text_encoder.parameters(), "lr": cfg_paras['text_encoder_lr']},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": cfg_paras['head_lr'], "weight_decay": cfg_paras['weight_decay']}
    ]
    
    run = neptune.init_run(
        project="ce-hou/LLM-CRIME",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
    )  # your credentials
    run["parameters"] = cfg_paras
        
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg_paras['patience'], factor=cfg_paras['factor']
    )
    step = "epoch"

    best_loss = float('inf')
    count_after_best = 0
    for epoch in range(cfg_paras['epochs']):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, cfg_paras)
        run["train/train_loss"].append(train_loss.avg)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, cfg_paras)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            count_after_best = 0
            torch.save(model.state_dict(), os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name']))
            print("Saved Best Model! to", os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name']))
        
        lr_scheduler.step(valid_loss.avg)
        run["train/valid_loss"].append(valid_loss.avg)
        
        count_after_best += 1
        if count_after_best > cfg_paras['early_stopping_threshold']:
            print("Early Stopping!")
            break


def main(cfg_paras):
    
    # CFG = Configurations(cfg_paras)
    dataset_path = cfg_paras['dataset_path']
    df = pd.read_pickle(dataset_path)
    
    # train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(cfg_paras['text_tokenizer'])
    
    train_num = int(len(df) * 0.7)
    train_loader = build_loaders(df[:train_num], tokenizer, mode="train", cfg_paras=cfg_paras)
    valid_loader = build_loaders(df[train_num:], tokenizer, mode="valid", cfg_paras=cfg_paras)

    print("use device: ", cfg_paras['device_step1'])
    model = CLIPModel(cfg_paras).to(cfg_paras['device'])
    params = [
        {"params": model.image_encoder.parameters(), "lr": cfg_paras['image_encoder_lr']},
        {"params": model.text_encoder.parameters(), "lr": cfg_paras['text_encoder_lr']},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": cfg_paras['head_lr'], "weight_decay": cfg_paras['weight_decay']}
    ]
    
    run = neptune.init_run(
        project="ce-hou/LLM-CRIME",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
    )  # your credentials
    run["parameters"] = cfg_paras
        
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg_paras['patience'], factor=cfg_paras['factor']
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(cfg_paras['epochs']):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, cfg_paras)
        run["train/train_loss"].append(train_loss.avg)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, cfg_paras)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            if not os.path.exists(os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name'])):
                os.makedirs(os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name']))
            torch.save(model.state_dict(), os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name']))
            print("Saved Best Model! to", os.path.join(cfg_paras['save_model_path'], cfg_paras['save_model_name']))
        
        lr_scheduler.step(valid_loss.avg)
        run["train/valid_loss"].append(valid_loss.avg)
        
# if __name__ == '__main__':
    # dataset_path = "/data_nas/cehou/LLM_safety/dataset_30_female_HongKong_murder_746.pkl"
    # dataset_config = dataset_path.split("/")[-1].split("_")
    # save_model_path = f"/data_nas/cehou/LLM_safety/model/model_{dataset_config[1]}_{dataset_config[2]}_{dataset_config[3]}_{dataset_config[4]}.pt"
    # main(cfg_paras)