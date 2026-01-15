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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
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

class DenseNet121(nn.Module):
    def __init__(self, pretrained=True, embedding_dim=256):
        super().__init__()

        base = models.densenet121(pretrained=pretrained)
        self.features = base.features
        self.features.add_module('relu', nn.ReLU(inplace=True))

        # 🔥 Attention
        self.attn = CBAM(channels=1024)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 🔥 BNNeck (rất quan trọng với Triplet)
        self.bnneck = nn.BatchNorm1d(1024)
        self.bnneck.bias.requires_grad_(False)

        self.fc = nn.Linear(1024, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x).flatten(1)

        x = self.bnneck(x)
        x = self.fc(x) # Fully connected layer

        x = F.normalize(x, dim=1) # L2 normalize
        return x

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