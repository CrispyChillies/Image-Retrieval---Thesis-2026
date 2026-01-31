import os
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet, VINDRDataSet

from loss import TripletMarginLoss
from sampler import PKSampler
from model import ResNet50, DenseNet121, ConvNeXtV2, HybridConvNeXtViT, ConceptCLIPBackbone

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

# Use 384x384 for ConvNeXtV2 and Hybrid model, 224x224 for others
img_size = 384
resize_size = 432 if img_size == 384 else 256

train_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                          transforms.Resize(resize_size),
                                          transforms.RandomResizedCrop(img_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                      transforms.Resize(resize_size),
                                      transforms.CenterCrop(img_size),
                                      transforms.ToTensor(),
                                      normalize])


train_dataset = VINDRDataSet(data_dir=os.path.join('', 'train_data'), csv_file='/home/aaronpham5504/Coding/Image-Retrieval---Thesis-2026/vindr/image_labels_train.csv', transform=train_transform)
test_dataset = VINDRDataSet(data_dir=os.path.join('', 'test_data'),csv_file='/home/aaronpham5504/Coding/Image-Retrieval---Thesis-2026/vindr/image_labels_test.csv', transform=test_transform)

targets = train_dataset.labels
train_loader = DataLoader(train_dataset, batch_size=16, sampler=PKSampler(targets, 4, 4))