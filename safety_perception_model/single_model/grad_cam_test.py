import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import sys
sys.path.append("/code/LLM-crime/single_model")
sys.path.append("/code/LLM-crime")
from LLM_feature_extractor import LLaVaFeatureExtractor
from safety_train_new import Extractor, Adaptor, Classifier, FullModel

from glob import glob
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
from itertools import product
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

root_path = "/data2/cehou/LLM_safety/LLM_models/safety_perception_model/only_img/multi_extractor_small_dataset_20241228/lr_1e-05_visual_feature_extractor_ViT_LLM_loaded_False/"
model_path = glob(f"{root_path}*.pt")[0]

parameters_path = glob(f"{root_path}*.csv")[0]
parameters = pd.read_csv(parameters_path)
parameters.drop(["Unnamed: 0"], axis=1, inplace=True)
parameters = parameters.iloc[0].to_dict()

def vision_model_test(model, test_loader, model_type='feature_extractor', LLM_model=None):
    if LLM_model is not None:
        LLM_pre_extractor = LLM_model
    model.eval()  # 切换到评估模式
    all_outputs = []
    all_labels = []
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for data, target in tqdm(test_loader):
            if LLM_model is not None:
                data = LLM_pre_extractor([data[i] for i in range(len(data))])
            data, target = data.cuda(), target.cuda().long()
            if model_type == 'feature_extractor':
                output = model(data)['features']
            else:
                output = model(data)
            # output = model(data)['features']
            all_outputs.append(output)
            all_labels.append(target)
    
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_outputs, all_labels

# data = pd.read_csv(parameters['placepulse_datapath'])
data = pd.read_pickle('/data2/cehou/LLM_safety/img_text_data/finished/dataset_30_male_HongKong_traffic accident_GSV_2000_all_4712.pkl')
data['label'] = 1
data.rename(columns={"panoid": "Image_ID"}, inplace=True)
error_panoid = {'_ZyueibIbaC6UyhG4WcRAA', 'kXQjqDhu9HnvAdIkKl-37Q'}
data = data[~data['Image_ID'].isin(set(error_panoid))]
data_ls = data[data['label'] != 0]
data_ls.loc[data_ls[data_ls['label'] == -1].index, 'label'] = 0
transform = get_transforms((224,224))
train_num = int(len(data_ls) * 0.6)
valid_num = int(len(data_ls) * 0.2)


LLM_pre_extractor = None
one_dataset = SafetyPerceptionDataset(data[:200], transform=transform, paras=parameters, SVI_type="GSV")
one_loader = torch.utils.data.DataLoader(one_dataset, batch_size=parameters['batch_size'])

total_dataset = SafetyPerceptionDataset(data, transform=transform, paras=parameters, SVI_type="GSV") 
total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=parameters['batch_size'])

extractor = Extractor(pretrained_model=parameters['visual_feature_extractor']) # [128, 512]
adaptor = Adaptor(input_dim=parameters['input_dim'], projection_dim=parameters['adaptor_output_dim'], data_type='image') # [128, 256]
classifier = Classifier(input_dim=parameters['adaptor_output_dim'], num_classes=parameters['num_classes']) # [128, 2]
model = FullModel(extractor, adaptor, classifier).cuda()

# Load the saved model parameters
model.load_state_dict(torch.load(model_path), strict=False)

feature_extractor = create_feature_extractor(model, return_nodes={'extractor.model.encoder.ln': 'features'})
# all_outputs, all_labels = vision_model_test(feature_extractor, one_loader, model_type='feature_extractor')
# valid_outputs, valid_labels = model_test(model, valid_loader)

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

target_layers = [extractor.model.encoder.ln]
input_tensor = next(iter(total_loader))[0]
rgb_img = input_tensor[0].permute(1, 2, 0).cpu().numpy()

# We have to specify the target we want to generate the CAM for.
targets = [ClassifierOutputTarget(1)]

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# Construct the CAM object once, and then re-use it on many images.
with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#   print(input_tensor.shape)
#   print(targets)
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  # You can also get the model outputs without having to redo inference
  model_outputs = cam.outputs