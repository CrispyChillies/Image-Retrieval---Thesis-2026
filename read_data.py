# encoding: utf-8

"""
Read images and corresponding labels.
"""

import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
# from segmentation import segment_and_mask
import numpy as np
import cv2
import pandas as pd


class ISICDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, use_melanoma=True, mask_dir=None, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            use_melanoma: whether or not to use melanoma samples (default = True).
            mask_dir: optional path to segmentation masks directory.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        mask_names = []
        with open(image_list_file, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for line in reader:
                image_name = line[0]+'.jpg'
                if float(line[1]) == 1:
                    label = 2  # melanoma
                elif float(line[2]) == 1:
                    label = 1  # seborrheic keratosis
                else:
                    label = 0  # nevia
                if label == 2 and use_melanoma is False:
                    continue
                if mask_dir is not None:
                    raise NotImplementedError
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.mask_names = mask_names
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new('RGB', image.size), mask)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, use_covid=True, mask_dir=None, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            use_covid: whether or not to use COVID-19 samples (default = True).
            mask_dir: optional path to segmentation masks directory.
            transform: optional transform to be applied on a sample.
        """
        mapping = {
            "normal": 0,
            "pneumonia": 1,
            "COVID-19": 2,
        }

        image_names = []
        labels = []
        mask_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[1]
                label = mapping[items[2]]
                if label == 2 and use_covid is False:
                    continue
                if mask_dir is not None:
                    mask_name = os.path.join(
                        mask_dir, os.path.splitext(image_name)[0] + '_xslor.png')
                    mask_names.append(mask_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.mask_names = mask_names
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new('RGB', image.size), mask)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
    def __len__(self):
        return len(self.image_names)


# TBX11k Dataset for retrieval (classification/retrieval only, no bbox)
class TBX11kDataSet(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the csv file (train.csv or test.csv).
            transform: optional transform to be applied on a sample.
        """
        self.image_names = []
        self.labels = []
        self.transform = transform

        # Map image_type to integer label
        # image_type: tb, healthy, sick_but_no_tb
        self.type_map = {"tb": 0, "healthy": 1, "sick_but_no_tb": 2}

        import csv
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["fname"]
                image_type = row["image_type"].strip()
                # Only use images with valid image_type
                if image_type not in self.type_map:
                    continue
                img_path = os.path.join(data_dir, fname)
                self.image_names.append(img_path)
                self.labels.append(self.type_map[image_type])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)


class VINDRDataSet(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir: Đường dẫn đến thư mục chứa ảnh (.png).
            csv_file: Đường dẫn đến file CSV (chứa image_id và các cột label).
            transform: Các phép biến đổi ảnh (Augmentation).
        """
        self.data_dir = data_dir
        self.transform = transform
        
        self.labels = [
            "Aortic enlargement", "Cardiomegaly", 
            "Pleural effusion", "Pleural thickening", 
            "Lung Opacity", "No finding"
        ]
        
        df = pd.read_csv(csv_file)
        self.data = df.groupby("image_id")[self.labels].max().reset_index()
        
        self.image_ids = self.data["image_id"].tolist()
        self.labels = self.data[self.labels].values

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.data_dir, f"{img_id}.png")
        
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.image_ids)

# if __name__ == "__main__":
#     dataset = VINDRDataSet(data_dir='', csv_file='/home/aaronpham5504/Coding/Image-Retrieval---Thesis-2026/vindr/image_labels_train.csv')
#     img, target = dataset[0]

#     print(f"Image shape: {img.size}")
#     print(f"Target vector: {target}")
#     print(f"Labels mapping: {dataset.target_columns}")