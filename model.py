import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import AutoModel


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

class SpatialAttentionMask(nn.Module):
    """
    Generate soft spatial attention mask m 
    """
    def __init__(self, in_channels, hidden_ratio=4):
        super().__init__()

        hidden = in_channels // hidden_ratio

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: attention mask m (B, 1, H, W)
        """
        return self.net(x)
    
class DenseNet121(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=256):
        super().__init__()

        base = models.densenet121(pretrained=pretrained)
        features = base.features

        # -------------------------
        # f1: low-mid level feature
        # -------------------------
        self.f1 = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
            features.denseblock1,
            features.transition1,
            features.denseblock2,
        )

        # -------------------------
        # f2: high-level feature
        # -------------------------
        self.f2 = nn.Sequential(
            features.transition2,
            features.denseblock3,
            features.transition3,
            features.denseblock4,
            features.norm5
        )

        # Channel dimension after denseblock2 = 512
        self.attention = SpatialAttentionMask(in_channels=512)

        # Pooling + embedding
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # BNNeck (metric learning)
        self.bnneck = nn.BatchNorm1d(1024)
        self.bnneck.bias.requires_grad_(False)

        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x, return_attention=False):
        """
        return_attention: dùng cho x-MIR visualization
        """
        # f1 feature
        f1 = self.f1(x)                     # (B, 512, H, W)

        # attention mask
        m = self.attention(f1)              # (B, 1, H, W)

        # f2 feature
        f2 = self.f2(f1)                    # (B, 1024, H', W')

        # resize mask if needed
        if m.shape[-2:] != f2.shape[-2:]:
            m = F.interpolate(
                m,
                size=f2.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # apply mask (GATING, NOT residual)
        f_att = f2 * m

        # global embedding
        z = self.avgpool(f_att).flatten(1)
        z = self.bnneck(z)
        z = self.fc(z)
        z = F.normalize(z, dim=1)

        if return_attention:
            return z, m
        return z
 
# class DenseNet121(nn.Module):
#     """Model modified.

#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.

#     """

#     def __init__(self, pretrained=True, embedding_dim=None):
#         super(DenseNet121, self).__init__()
#         # load pretrained model
#         self.densenet121 = models.densenet121(pretrained=pretrained)
#         # remove classifier
#         self.densenet121 = nn.Sequential(
#             *list(self.densenet121.children())[:-1])

#         # add ReLU and average pooling
#         self.densenet121[0].add_module('relu', nn.ReLU(inplace=True))
#         self.densenet121.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

#         # Assumes batchnorm layer is present
#         in_features = self.densenet121[0].norm5.num_features
#         self.fc = nn.Linear(
#             in_features, embedding_dim) if embedding_dim else None

#     def forward(self, x):
#         # extract features
#         x = self.densenet121(x)
#         x = torch.flatten(x, 1)
#         if self.fc:
#             x = self.fc(x)
#         # normalize features
#         x = F.normalize(x, dim=1)
#         return x

class MedSigLIPGoogle(nn.Module):
    def __init__(self, model_name="google/medsiglip-448"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # embedding dùng cho retrieval
        self.embedding_dim = self.model.config.projection_dim

        # optional: freeze text tower
        if hasattr(self.model, "text_model"):
            for p in self.model.text_model.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, 448, 448), normalized [-1, 1]
        """
        outputs = self.model(pixel_values=x)

        embeds = outputs.image_embeds  # <<< QUAN TRỌNG
        embeds = F.normalize(embeds, dim=-1)

        return embeds