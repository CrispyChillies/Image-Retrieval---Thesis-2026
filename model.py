import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoProcessor


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=None):
        super(ResNet50, self).__init__()
        # load pretrained model
        self.resnet50 = models.resnet50(pretrained=pretrained)
        # remove classifier
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Assumes batchnorm layer is present
        in_features = self.resnet50[7][2].bn3.num_features
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        # extract features
        x = self.resnet50(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        # normalize features
        x = F.normalize(x, dim=1)
        return x


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, pretrained=True, embedding_dim=None):
        super(DenseNet121, self).__init__()
        # load pretrained model
        self.densenet121 = models.densenet121(pretrained=pretrained)
        # remove classifier
        self.densenet121 = nn.Sequential(
            *list(self.densenet121.children())[:-1])

        # add ReLU and average pooling
        self.densenet121[0].add_module('relu', nn.ReLU(inplace=True))
        self.densenet121.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

        # Assumes batchnorm layer is present
        in_features = self.densenet121[0].norm5.num_features
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        # extract features
        x = self.densenet121(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        # normalize features
        x = F.normalize(x, dim=1)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXtV2 model for feature extraction.

    Uses ConvNeXtV2 Base from timm with optional embedding layer.
    """

    def __init__(self, pretrained=True, embedding_dim=None):
        super(ConvNeXtV2, self).__init__()
        # load pretrained model from timm
        self.convnext = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0  # removes classifier, returns features directly
        )
        
        # get the number of input features
        in_features = self.convnext.num_features
        
        # optional embedding layer
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        # extract features
        x = self.convnext(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        # normalize features
        x = F.normalize(x, dim=1)
        return x


class SwinV2(nn.Module):
    # SwinV2 model for feature extraction using timm

    def __init__(self, pretrained=True, embedding_dim=None):
        super(SwinV2, self).__init__()
        # load pretrained model from timm
        self.swinv2 = timm.create_model(
            'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=0  # removes classifier, returns features directly
        )
        
        # get the number of input features
        in_features = self.swinv2.num_features
        
        # optional embedding layer
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None
    
    def forward(self, x):
        # extract features
        x = self.swinv2(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        # normalize features
        x = F.normalize(x, dim=1)
        return x

class MedSigLIP(nn.Module):
    def __init__(self, model_name="google/medsiglip-448", embed_dim=512, unfreeze_layers=2):
        super().__init__()
        full_model = AutoModel.from_pretrained(model_name)
        self.backbone = full_model.vision_model 
        
        # --- BƯỚC 1: Đóng băng toàn bộ Backbone ---
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # --- BƯỚC 2: Mở khóa (Unfreeze) các lớp cuối ---
        # MedSigLIP ViT thường có 24 layers (0-23)
        # Chúng ta sẽ mở khóa các layer cuối cùng và lớp LayerNorm sau cùng
        if unfreeze_layers > 0:
            # Mở khóa các transformer blocks cuối
            for layer in self.backbone.encoder.layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Mở khóa post_layernorm (quan trọng để cân chỉnh embedding cuối)
            for param in self.backbone.post_layernorm.parameters():
                param.requires_grad = True

        # --- BƯỚC 3: Projection Head luôn luôn được train ---
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output 
        embeddings = self.projection(features)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# Use transformer Automodel for ConceptCLIP using JerrryNie/ConceptCLI
# Image Encoder: SigLIP-ViT-400M-16
# Text Encoder: PubMedBERT
class conceptCLIP(nn.Module):
    def __init__(self, model_name='JerrryNie/ConceptCLIP', embedding_dim=None):
        super(conceptCLIP, self).__init__()
        # load pretrained model from transformers
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
