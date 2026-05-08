import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attention)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.apply(self._init_weights)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return F.relu(out + identity, inplace=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)


class ATHNet(nn.Module):
    def __init__(self, hash_size, num_classes, input_size=256):
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError("input_size must be divisible by 8 for ATHNet.")

        final_size = input_size // 8
        flattened_dim = final_size * final_size

        self.net1 = nn.Sequential(
            ResBlock(in_channels=3, out_channels=16, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.sa = SpatialAttention()
        self.net2 = nn.Sequential(
            ResBlock(in_channels=16, out_channels=8, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.dense = ResBlock(in_channels=8, out_channels=1, stride=2)
        self.hashlayer = nn.Linear(flattened_dim, hash_size)
        self.typelayer = nn.Linear(flattened_dim, num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.net1(x)
        x = self.sa(x) * x
        x = self.net2(x)
        x = self.dense(x)
        x = torch.flatten(x, 1)
        hash_codes = self.hashlayer(x)
        logits = self.typelayer(x)
        return hash_codes, logits

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)


class TripletHashLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, anchor_hash, positive_hash, negative_hash):
        margin_val = self.margin * anchor_hash.shape[1]
        pos_loss = torch.mean(self.mse_loss(anchor_hash, positive_hash), dim=1)
        neg_loss = torch.mean(self.mse_loss(anchor_hash, negative_hash), dim=1)
        zeros = torch.zeros_like(neg_loss)
        loss = torch.maximum(zeros, margin_val - neg_loss + pos_loss)
        return loss.mean()
