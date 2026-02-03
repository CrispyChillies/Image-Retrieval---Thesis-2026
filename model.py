import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm

from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


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
    """
    Model modified.

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


class ChannelAttention(nn.Module):
    """Channel Attention module for MXA Block"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Channel attention
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention module for MXA Block"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return x * self.sigmoid(out)


class HybridConvNeXtViT(nn.Module):
    """
    Hybrid ConvNeXtV2-ViT Architecture with Multi-scale Attention (MXA) Block.
    
    Features:
    - ConvNeXtV2-Base backbone with MXA Block (Channel + Spatial Attention)
    - ViT-B/16 backbone with Token Pooling  
    - Feature Fusion module
    - Global Average Pooling for final feature extraction
    """
    
    def __init__(self, pretrained=True, embedding_dim=1024):
        super(HybridConvNeXtViT, self).__init__()
        
        # ConvNeXtV2 Branch
        self.convnext = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0,
            features_only=True  # Get intermediate feature maps
        )
        convnext_features = 1024  # ConvNeXtV2-Base last stage channels
        
        # ViT Branch
        self.vit = timm.create_model(
            'vit_base_patch16_384',
            pretrained=pretrained,
            num_classes=0
        )
        vit_features = self.vit.num_features  # 768
        
        # MXA Block for ConvNeXt (Multi-scale Attention)
        self.channel_attention = ChannelAttention(convnext_features)
        self.spatial_attention = SpatialAttention()
        
        # Feature Fusion
        fused_features = convnext_features + vit_features  # 1024 + 768 = 1792
        self.fusion = nn.Sequential(
            nn.Linear(fused_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final embedding layer
        self.fc = nn.Linear(embedding_dim, embedding_dim) if embedding_dim else None
        
    def forward(self, x):
        # ConvNeXt Branch with MXA
        # Get feature maps from last stage
        conv_features_list = self.convnext(x)
        conv_features = conv_features_list[-1]  # Get last stage features [B, 1024, H, W]
        
        # Apply MXA Block (Channel + Spatial Attention)
        conv_features = self.channel_attention(conv_features)
        conv_features = self.spatial_attention(conv_features)
        
        # Global Average Pooling
        conv_features = F.adaptive_avg_pool2d(conv_features, (1, 1))
        conv_features = torch.flatten(conv_features, 1)
        
        # ViT Branch with Token Pooling
        vit_features = self.vit.forward_features(x)
        # Extract CLS token (first token)
        if len(vit_features.shape) == 3:
            vit_features = vit_features[:, 0]  # Take CLS token
        
        # Feature Fusion
        fused = torch.cat([conv_features, vit_features], dim=1)
        fused = self.fusion(fused)
        
        if self.fc:
            fused = self.fc(fused)
        
        # Normalize features
        fused = F.normalize(fused, dim=1)
        
        return fused

class ConceptCLIPBackbone(nn.Module):
    """
    ConceptCLIP Backbone for Medical Image Retrieval.
    
    Wraps the ConceptCLIP model from HuggingFace (JerrryNie/ConceptCLIP)
    with optional projection head for embedding learning.
    
    Features:
    - Pre-trained on medical images
    - Vision encoder based on ViT architecture
    - Optional embedding dimension projection
    - Supports freezing backbone for transfer learning
    """
    
    def __init__(
        self,
        pretrained=True,
        embedding_dim=None,         # optional projection to custom embedding dim
        freeze=False,               # whether to freeze ConceptCLIP weights
        processor_normalize=True    # use ConceptCLIP's processor normalization
    ):
        super().__init__()
        
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers library is required for ConceptCLIP. "
                "Install it with: pip install transformers"
            )
        
        # Load ConceptCLIP model
        if pretrained:
            self.model = AutoModel.from_pretrained(
                'JerrryNie/ConceptCLIP',
                trust_remote_code=True
            )
        else:
            raise NotImplementedError("ConceptCLIP requires pretrained weights")
        
        self.processor_normalize = processor_normalize
        
        # ConceptCLIP vision encoder output dimension (usually 768 or 512)
        # We'll determine it dynamically on first forward pass
        self.out_dim = None
        
        # Optional projection head
        self._embedding_dim = embedding_dim
        self.fc = None  # will be initialized after first forward pass
        
        # Denormalization parameters for ImageNet normalization
        # (ConceptCLIP processor expects [0, 1] range, but dataloader normalizes)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Optionally freeze backbone
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
    
    def denormalize(self, x):
        """Denormalize images from ImageNet normalization to [0, 1] range."""
        return x * self.std + self.mean
    
    def forward(self, x):
        """
        Forward pass through ConceptCLIP vision encoder.
        
        Args:
            x: tensor [B, 3, H, W] - normalized images from dataloader
        
        Returns:
            embeddings: tensor [B, embedding_dim] - normalized feature embeddings
        """
        # Denormalize images for ConceptCLIP processor
        x_denorm = self.denormalize(x)
        
        # Clip to [0, 1] range (safety measure)
        x_denorm = torch.clamp(x_denorm, 0.0, 1.0)
        
        # ConceptCLIP expects list of tensors or processor format
        # For efficiency, we'll process as batch
        if self.processor_normalize:
            # Use ConceptCLIP's processor (may apply additional normalization)
            # Convert batch to list for processor
            images_list = [img for img in x_denorm]
            inputs = self.processor(
                images=images_list,
                return_tensors='pt',    
                padding=True
            ).to(x.device)
        else:
            # Direct forward without processor (use denormalized images)
            inputs = {'pixel_values': x_denorm}
        
        # Extract image features
        outputs = self.model(**inputs)
        features = outputs['image_features']  # [B, hidden_dim]
        
        # Initialize projection layer on first forward pass
        if self.out_dim is None:
            self.out_dim = features.shape[-1]
            if self._embedding_dim is not None:
                self.fc = nn.Linear(self.out_dim, self._embedding_dim).to(features.device)
        
        # Apply optional projection
        if self.fc is not None:
            features = self.fc(features)
        
        # Normalize features (L2 normalization)
        features = F.normalize(features, dim=1)
        
        return features

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelWiseAverage(nn.Module):
    def forward(self, x):
        # Average over channel dimension, keep spatial
        return torch.mean(x, dim=1, keepdim=True)

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)

class Resnet50_with_Attention(nn.Module):
    def __init__(self, embedding_dim=64):
        super(Resnet50_with_Attention, self).__init__()
        resnet = resnet50(pretrained=True)
        # f1: up to layer3 (conv3_4)
        self.f1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        # f2: layer4
        self.f2 = resnet.layer4

        # Attention branch
        # Use a bottleneck block (from ResNet), SE, channel-wise avg, sigmoid
        # Use the last block of layer3 for bottleneck
        bottleneck_channels = 1024  # layer3 output channels
        self.attention_module = nn.Sequential(
            Bottleneck(1024, 256),
            SEBlock(bottleneck_channels),
            ChannelWiseAverage(),
            nn.Sigmoid()
        )

        # Projection head
        self.g = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            L2Norm()
        )

    def forward(self, x):
        feat_mid = self.f1(x)  # [B, 1024, H, W]
        mask = self.attention_module(feat_mid)  # [B, 1, H, W]
        feat_deep = self.f2(feat_mid)           # [B, 2048, H, W]

        if mask.shape[-2:] != feat_deep.shape[-2:]:
            mask = F.interpolate(mask, size=feat_deep.shape[-2:], mode='bilinear', align_corners=False)
        localized_feat = feat_deep * mask

        final_embedding = self.g(localized_feat)
        return final_embedding, mask