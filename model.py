import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel, AutoProcessor


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=None, num_labels=None):
        super(ResNet50, self).__init__()
        # load pretrained model
        self.resnet50 = models.resnet50(pretrained=pretrained)
        # remove classifier
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        # Assumes batchnorm layer is present
        in_features = self.resnet50[7][2].bn3.num_features
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None
        output_features = embedding_dim if embedding_dim else in_features
        self.classification_head = (
            nn.Linear(output_features, num_labels) if num_labels else None
        )

    def forward(self, x):
        # extract features
        x = self.resnet50(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        if self.classification_head is not None:
            return {
                "embedding": F.normalize(x, dim=1),
                "logits": self.classification_head(x),
            }
        # normalize features
        x = F.normalize(x, dim=1)
        return x


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, pretrained=True, embedding_dim=None, num_labels=None):
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
        output_features = embedding_dim if embedding_dim else in_features
        self.classification_head = (
            nn.Linear(output_features, num_labels) if num_labels else None
        )

    def forward(self, x):
        # extract features
        x = self.densenet121(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        if self.classification_head is not None:
            return {
                "embedding": F.normalize(x, dim=1),
                "logits": self.classification_head(x),
            }
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


class SRA(nn.Module):
    """Spatial Residual Attention for retrieval.

    Uses K attention heads to compute spatially-attended features,
    then averages them with GAP features to preserve the original
    backbone feature dimension (e.g., 1024 for ConvNeXtV2-Base).
    """

    def __init__(self, input_dim, num_heads=8, lam=0.1, norm_layer=None):
        super().__init__()
        self.num_heads = num_heads
        self.lam = lam
        self.norm_layer = norm_layer

        # attention: K heads, each producing a spatial attention map
        self.conv_att = nn.Conv2d(input_dim, num_heads, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        nn.init.normal_(self.conv_att.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        b, c, h, w = x.size()

        # Baseline branch mirrors ConvNeXtV2's pretrained retrieval head.
        gap_feat = torch.mean(x, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        if self.norm_layer is not None:
            gap_feat = self.norm_layer(gap_feat)
        gap_feat = torch.flatten(gap_feat, 1)  # (B, C)

        # attention-weighted branch: K heads
        att_map = self.conv_att(x).view(b, self.num_heads, h * w)  # (B, K, H*W)
        att_score = self.softmax(att_map)

        x_flat = x.view(b, c, h * w)  # (B, C, H*W)
        # (B, K, H*W) x (B, H*W, C) → (B, K, C)
        csra_feat = torch.bmm(att_score, x_flat.permute(0, 2, 1))
        # average across heads → (B, C)
        csra_feat = csra_feat.mean(dim=1).view(b, c, 1, 1)
        if self.norm_layer is not None:
            csra_feat = self.norm_layer(csra_feat)
        csra_feat = torch.flatten(csra_feat, 1)

        # residual fusion: GAP + λ * attention-weighted features → (B, C)
        return gap_feat + self.lam * csra_feat


class ConvNeXtV2_SRA(nn.Module):
    """ConvNeXtV2 with SRA (Spatial Residual Attention) head for retrieval.

    Uses spatial feature maps from ConvNeXtV2 backbone and applies SRA
    with multiple attention heads. Outputs same dimension as backbone (1024).
    """

    def __init__(self, pretrained=True, num_heads=8, lam=0.1):
        super(ConvNeXtV2_SRA, self).__init__()
        # load pretrained model from timm
        self.convnext = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0
        )

        in_features = self.convnext.num_features
        self.sra = SRA(
            in_features,
            num_heads=num_heads,
            lam=lam,
            norm_layer=self.convnext.head.norm,
        )

    def forward(self, x):
        # extract spatial feature maps (B, C, H, W) before pooling
        x = self.convnext.forward_features(x)
        # SRA produces attention-refined features (B, C)
        x = self.sra(x)
        # normalize for retrieval
        x = F.normalize(x, dim=1)
        return x


class PCAMPool(nn.Module):
    """Probabilistic-CAM pooling adapted for retrieval embeddings."""

    def __init__(
        self,
        input_dim,
        num_classes,
        lam=0.1,
        norm_layer=None,
        embedding_dim=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.norm_layer = norm_layer
        self.classifier = nn.Conv2d(input_dim, num_classes, kernel_size=1)
        self.fc = nn.Linear(input_dim, embedding_dim) if embedding_dim else None

    def forward(self, x):
        b, c, h, w = x.size()

        gap_feat = torch.mean(x, dim=(2, 3), keepdim=True)
        if self.norm_layer is not None:
            gap_feat = self.norm_layer(gap_feat)
        gap_feat = torch.flatten(gap_feat, 1)

        x_for_pcam = self.norm_layer(x) if self.norm_layer is not None else x
        cam_logits = self.classifier(x_for_pcam)  # (B, num_classes, H, W)
        pcam_probs = torch.sigmoid(cam_logits)
        pcam_weights = pcam_probs.view(b, self.num_classes, h * w)
        pcam_weights = pcam_weights / (pcam_weights.sum(dim=2, keepdim=True) + 1e-8)

        x_flat = x_for_pcam.view(b, c, h * w)
        class_pooled = torch.bmm(pcam_weights, x_flat.permute(0, 2, 1))

        classifier_weight = self.classifier.weight.view(self.num_classes, c)
        class_logits = torch.einsum("bkc,kc->bk", class_pooled, classifier_weight)
        if self.classifier.bias is not None:
            class_logits = class_logits + self.classifier.bias

        class_weights = F.softmax(class_logits, dim=1).unsqueeze(2)
        pcam_feat = torch.sum(class_weights * class_pooled, dim=1)
        feat = gap_feat + self.lam * pcam_feat

        if self.fc is not None:
            feat = self.fc(feat)

        embedding = F.normalize(feat, dim=1)
        return embedding, class_logits, pcam_probs


class ConvNeXtV2_PCAM(nn.Module):
    """ConvNeXtV2 with PCAM pooling for retrieval."""

    def __init__(self, pretrained=True, num_classes=3, lam=0.1, embedding_dim=None):
        super(ConvNeXtV2_PCAM, self).__init__()
        self.convnext = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k_384',
            pretrained=pretrained,
            num_classes=0
        )

        self.pcam = PCAMPool(
            self.convnext.num_features,
            num_classes=num_classes,
            lam=lam,
            norm_layer=self.convnext.head.norm,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        x = self.convnext.forward_features(x)
        embedding, class_logits, pcam_probs = self.pcam(x)
        if self.training:
            return {
                "embedding": embedding,
                "class_logits": class_logits,
                "pcam_maps": pcam_probs,
            }
        return embedding


class ConvNeXtV2_DinoDistill(nn.Module):
    """ConvNeXtV2 student with an online DINOv2 teacher for retrieval distillation."""

    def __init__(
        self,
        pretrained=True,
        embedding_dim=None,
        dinov2_model_name="vit_base_patch14_dinov2.lvd142m",
        teacher_trainable=False,
        unfreeze_blocks=0,
    ):
        super(ConvNeXtV2_DinoDistill, self).__init__()
        self.student = ConvNeXtV2(pretrained=pretrained, embedding_dim=embedding_dim)
        self.teacher = DinoV2(
            model_name=dinov2_model_name,
            pretrained=pretrained,
            embedding_dim=None,
            unfreeze_blocks=unfreeze_blocks if teacher_trainable else 0,
        )
        self.teacher_trainable = teacher_trainable
        self.teacher_input_size = None
        patch_embed = getattr(self.teacher.backbone, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "img_size"):
            self.teacher_input_size = patch_embed.img_size
        if not teacher_trainable:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if not self.teacher_trainable:
            self.teacher.eval()
        return self

    def forward(self, x):
        student_embedding = self.student(x)
        if self.training:
            teacher_x = x
            if self.teacher_input_size is not None:
                teacher_x = F.interpolate(
                    x,
                    size=self.teacher_input_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if self.teacher_trainable:
                teacher_embedding = self.teacher(teacher_x)
            else:
                with torch.no_grad():
                    teacher_embedding = self.teacher(teacher_x)
            return {
                "embedding": student_embedding,
                "teacher_embedding": teacher_embedding,
            }
        return student_embedding


class RadDinoTeacher(nn.Module):
    """Frozen RAD-DINO teacher that consumes the student's normalized tensor batch."""

    def __init__(self, model_name="microsoft/rad-dino", pretrained=True):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        self.input_size = (518, 518)
        self.register_buffer(
            "student_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "student_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rad_mean",
            torch.tensor([0.5307, 0.5307, 0.5307]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rad_std",
            torch.tensor([0.2583, 0.2583, 0.2583]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        x = x * self.student_std + self.student_mean
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(
            x,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.rad_mean) / self.rad_std
        outputs = self.model(pixel_values=x)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embedding, dim=1)


class ConvNeXtV2_RadDinoDistill(nn.Module):
    """ConvNeXtV2 student with a frozen RAD-DINO teacher for retrieval distillation."""

    def __init__(
        self,
        pretrained=True,
        embedding_dim=None,
        teacher_model_name="microsoft/rad-dino",
    ):
        super(ConvNeXtV2_RadDinoDistill, self).__init__()
        self.student = ConvNeXtV2(pretrained=pretrained, embedding_dim=embedding_dim)
        self.teacher = RadDinoTeacher(model_name=teacher_model_name, pretrained=pretrained)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x):
        student_embedding = self.student(x)
        if self.training:
            with torch.no_grad():
                teacher_embedding = self.teacher(x)
            return {
                "embedding": student_embedding,
                "teacher_embedding": teacher_embedding,
            }
        return student_embedding


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


class DinoV2(nn.Module):
    """DINOv2 retrieval backbone with partial fine-tuning of the last blocks."""

    def __init__(
        self,
        model_name="vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        embedding_dim=None,
        unfreeze_blocks=3,
    ):
        super(DinoV2, self).__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError(
                f"DINOv2 backbone '{model_name}' does not expose transformer blocks."
            )

        unfreeze_blocks = max(0, min(unfreeze_blocks, len(blocks)))
        if unfreeze_blocks > 0:
            for block in blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        in_features = self.backbone.num_features
        self.fc = nn.Linear(
            in_features, embedding_dim) if embedding_dim else None

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        if self.fc:
            x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

def _convert_sdpa_to_eager_attention(model):
    """
    Forcibly replace SDPA attention modules with eager attention.
    This is necessary because SDPA attention cannot return attention weights.
    """
    try:
        from transformers.models.siglip.modeling_siglip import (
            SiglipAttention, SiglipSdpaAttention
        )
    except ImportError:
        # Older transformers version - try alternative import
        try:
            from transformers.models.siglip.modeling_siglip import SiglipAttention
            SiglipSdpaAttention = None
        except ImportError:
            return  # Can't patch, hopefully config settings work
    
    for name, module in model.named_modules():
        # Check if this is an SDPA attention module that needs replacement
        if SiglipSdpaAttention is not None and isinstance(module, SiglipSdpaAttention):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Create eager attention with same config
            config = module.config if hasattr(module, 'config') else model.config
            eager_attn = SiglipAttention(config)
            
            # Copy weights from SDPA to eager
            eager_attn.load_state_dict(module.state_dict(), strict=False)
            
            # Replace the module
            setattr(parent, attr_name, eager_attn)


class MedSigLIP(nn.Module):
    def __init__(self, model_name="google/medsiglip-448", embed_dim=512, unfreeze_layers=2):
        super().__init__()
        # Load with eager attention to enable attention weight extraction
        # Note: attn_implementation must be set BEFORE model instantiation
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        # Force eager attention at config level (required for attention output)
        config._attn_implementation = "eager"
        config._attn_implementation_autoset = False  # prevent auto-override
        config.output_attentions = True  # Enable attention output in config
        if hasattr(config, 'vision_config'):
            config.vision_config._attn_implementation = "eager"
            config.vision_config._attn_implementation_autoset = False
            config.vision_config.output_attentions = True
        
        full_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation='eager',
        )
        self.backbone = full_model.vision_model 
        
        # Ensure config is properly set on the backbone
        self.backbone.config._attn_implementation = "eager"
        self.backbone.config._attn_implementation_autoset = False
        self.backbone.config.output_attentions = True
        
        # Force convert any SDPA attention modules to eager (belt-and-suspenders)
        _convert_sdpa_to_eager_attention(self.backbone)
        
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

    def ensure_eager_attention(self):
        """
        Call this after loading weights to ensure attention can be extracted.
        Converts any SDPA attention modules to eager attention.
        """
        self.backbone.config._attn_implementation = "eager"
        self.backbone.config._attn_implementation_autoset = False
        self.backbone.config.output_attentions = True
        _convert_sdpa_to_eager_attention(self.backbone)
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to ensure eager attention is applied after loading weights."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Re-apply eager attention conversion after loading weights
        self.ensure_eager_attention()
        return result
    
    def verify_attention_output(self, device='cuda'):
        """
        Test that attention weights can be extracted.
        Returns True if attention output works, False otherwise.
        """
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448, device=device)
            outputs = self.backbone(
                pixel_values=dummy_input,
                output_attentions=True,
                return_dict=True
            )
            if outputs.attentions is None or len(outputs.attentions) == 0:
                return False
            # Check first layer has valid attention
            attn = outputs.attentions[0]
            return attn is not None and attn.numel() > 0

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
            text_transformer = self.model.text.transformer
            if hasattr(text_transformer, 'resblocks'):
                # OpenCLIP custom text transformer
                text_layers = text_transformer.resblocks
            elif hasattr(text_transformer, 'encoder') and hasattr(text_transformer.encoder, 'layer'):
                # HuggingFace BertModel (PubMedBERT) used as text encoder
                text_layers = text_transformer.encoder.layer
            else:
                text_layers = None
            if text_layers is not None:
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
        # IMPORTANT: return raw learnable parameters (log-space), NOT the model
        # output which already applies exp(). The loss functions apply exp().
        logit_scale = self.model.logit_scale if hasattr(self.model, 'logit_scale') else torch.tensor(0.0)
        logit_bias = self.model.logit_bias if hasattr(self.model, 'logit_bias') else torch.tensor(0.0)
        return {
            'image_features': outputs['image_features'],         # (B, D)
            'text_features': outputs['text_features'],           # (B, D) or (num_texts, D)
            'image_token_features': outputs.get('image_token_features', None),  # (B, N, D)
            'logit_scale': logit_scale,   # raw log-space parameter
            'logit_bias': logit_bias,     # raw bias parameter
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
