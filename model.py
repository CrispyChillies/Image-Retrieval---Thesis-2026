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

# Use transformer Automodel for ConceptCLIP using JerrryNie/ConceptCLIP
# Image Encoder: SigLIP-ViT-SO400M-14 (image_size=384, patch_size=14, 27×27=729 patches)
# Text Encoder: PubMedBERT
# Supports: IT-Align (image-text contrastive) + RC-Align (region-concept alignment)
class conceptCLIP(nn.Module):
    def __init__(self, model_name='JerrryNie/ConceptCLIP', embedding_dim=None,
                 unfreeze_vision_layers=4, unfreeze_text_layers=2):
        super(conceptCLIP, self).__init__()
        # Load the full ConceptCLIP model (OpenCLIP-based dual encoder)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # --- Freeze strategy: freeze most, unfreeze last N layers ---
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last N vision encoder layers
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'):
            # OpenCLIP ViT structure
            vision_layers = self.model.visual.transformer.resblocks
            for layer in vision_layers[-unfreeze_vision_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze vision ln_post and proj if present
            if hasattr(self.model.visual, 'ln_post'):
                for param in self.model.visual.ln_post.parameters():
                    param.requires_grad = True
            if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                self.model.visual.proj.requires_grad = True
        elif hasattr(self.model, 'vision_model'):
            # HF-style vision model
            if hasattr(self.model.vision_model, 'encoder'):
                vision_layers = self.model.vision_model.encoder.layers
                for layer in vision_layers[-unfreeze_vision_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            if hasattr(self.model.vision_model, 'post_layernorm'):
                for param in self.model.vision_model.post_layernorm.parameters():
                    param.requires_grad = True

        # Unfreeze last N text encoder layers
        if hasattr(self.model, 'text') and hasattr(self.model.text, 'transformer'):
            # OpenCLIP text structure
            text_layers = self.model.text.transformer.resblocks
            for layer in text_layers[-unfreeze_text_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif hasattr(self.model, 'text_model'):
            if hasattr(self.model.text_model, 'encoder'):
                text_layers = self.model.text_model.encoder.layer
                for layer in text_layers[-unfreeze_text_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        # Unfreeze logit_scale and logit_bias (learnable temperature)
        if hasattr(self.model, 'logit_scale'):
            self.model.logit_scale.requires_grad = True
        if hasattr(self.model, 'logit_bias'):
            self.model.logit_bias.requires_grad = True

        # Optional projection head for embedding_dim override
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            # Detect the hidden size from model config
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'projection_dim'):
                in_features = self.model.config.projection_dim
            elif hasattr(self.model, 'visual') and hasattr(self.model.visual, 'output_dim'):
                in_features = self.model.visual.output_dim
            else:
                in_features = 1152  # fallback (SO400M embed_dim)
            self.fc = nn.Linear(in_features, embedding_dim)
        else:
            self.fc = None

    def encode_image(self, pixel_values):
        """Encode images, returns (image_features [CLS], image_token_features [patches])"""
        outputs = self.model(pixel_values=pixel_values)
        image_features = outputs.get('image_features', None)  # (B, D) - CLS/pooled
        image_token_features = outputs.get('image_token_features', None)  # (B, N_patches, D)
        return image_features, image_token_features

    def encode_text(self, input_ids, attention_mask=None):
        """Encode text, returns (text_features [CLS], all token embeddings)"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.get('text_features', None)  # (B, D) - CLS/pooled
        return text_features

    def forward_clip(self, pixel_values, input_ids, attention_mask=None):
        """Full CLIP forward: encode both image and text, return all features.
        Used during ConceptCLIP fine-tuning with IT-Align + RC-Align.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return {
            'image_features': outputs['image_features'],         # (B, D)
            'text_features': outputs['text_features'],           # (B, D) or (num_texts, D)
            'image_token_features': outputs.get('image_token_features', None),  # (B, N, D)
            'logit_scale': outputs.get('logit_scale', torch.tensor(1.0)),
            'logit_bias': outputs.get('logit_bias', torch.tensor(0.0)),
        }

    def forward(self, pixel_values):
        """Image-only forward for retrieval evaluation and compatibility with existing pipeline.
        Returns L2-normalized image embeddings.
        """
        outputs = self.model(pixel_values=pixel_values)
        image_features = outputs['image_features']  # (B, D)
        if self.fc is not None:
            image_features = self.fc(image_features)
        return F.normalize(image_features, dim=1)
