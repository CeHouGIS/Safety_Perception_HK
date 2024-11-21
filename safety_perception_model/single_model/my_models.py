import torch
import torch.nn as nn
import timm
from torchvision import models, transforms

    
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout):
        super(TransformerRegressionModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)  # Adjust input_dim to match flattened image size
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # print("==============================")
        batch_size = x.size(0) 
        x = x.view(batch_size, -1)  # Flatten the image tensor, torch.Size([32, 360000])
        # print(x.shape)
        x = self.input_projection(x) # torch.Size([32, 512])
        # print(x.shape)
        x = x.unsqueeze(0)  # Add sequence dimension for transformer, torch.Size([1, 32, 512])
        # print(x.shape)
        x = self.transformer(x, x)  # torch.Size([1, 32, 512])
        # print(x.shape)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_projection(x) # torch.Size([32, 6])
        # print(x.shape)
        # print("==============================")
        return x
    
class ResNet50Model(nn.Module):
    def __init__(self, output_dim):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)

class FeatureResNet50(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super(FeatureResNet50, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        # Modify the first convolutional layer to accept 1-channel input
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the fully connected layer to match the input dimension and number of classes
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, self.num_classes)
    def forward(self, x):
        # Reshape the input to match the expected input shape of ResNet50
        x = x.view(-1, 1, 16, 16)  # Assuming input_dim=256, reshape to (batch_size, 1, 16, 16)
        x = self.resnet50(x)
        return x

class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class ViTClassifier(nn.Module):
    def __init__(self, output_dim):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, output_dim)  # 二分类，输出一个值

    def forward(self, x):
        return self.model(x)

class FeatureViTClassifier(nn.Module):
    def __init__(self, output_dim):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, output_dim)  # 二分类，输出一个值

    def forward(self, x):
        return self.model(x)