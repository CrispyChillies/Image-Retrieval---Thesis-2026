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


# VINDR Dataset for 4-class classification
class VINDRDataSet(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the csv file (should have image file names and labels).
            transform: optional transform to be applied on a sample.
        """
        self.image_names = []
        self.labels = []
        self.transform = transform

        # Map label string to integer
        self.label_map = {
            "Pneumonia": 0,
            "Tuberculosis": 1,
            "Other diseases": 2,
            "No finding": 3
        }

        import csv
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Adjust these keys if your CSV uses different column names
                fname = row.get("image_id") or row.get("fname") or row[list(row.keys())[0]]
                # Ensure .png extension for VINDR images
                if not fname.lower().endswith('.png'):
                    fname = fname + '.png'
                label = row.get("label") or row.get("finding") or row[list(row.keys())[1]]
                if label not in self.label_map:
                    label = "Other diseases"
                img_path = os.path.join(data_dir, fname)
                self.image_names.append(img_path)
                self.labels.append(self.label_map[label])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)

# Test main function for ChestXrayDataSet
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     # Example usage: update these paths as needed
#     data_dir = "./samples"  # directory containing images
#     image_list_file = "./train_split.txt"  # file listing images and labels
#     dataset = ChestXrayDataSet(data_dir, image_list_file, use_covid=True, mask_dir=None, transform=None)
#     print(f"Total images in dataset: {len(dataset)}")
#     # Load a sample image
#     img, label = dataset[0]
#     print(f"Sample label: {label}")
#     # Show the image
#     plt.imshow(img)
#     plt.title(f"Label: {label}")
#     plt.axis('off')
#     plt.show()

if __name__ == "__main__":
    # Example usage for VINDRDataSet
    data_dir = "/kaggle/input/vindr-cxr-physionet/train_data/train"  # Update to your images directory
    csv_file = "/kaggle/input/labeling-files-for-vindr/image_labels_test.csv"  # Update to your CSV path
    dataset = VINDRDataSet(data_dir, csv_file)
    print(f"Total images in dataset: {len(dataset)}")

    # Count samples per class
    from collections import Counter
    label_counts = Counter(dataset.labels)
    class_names = ["Pneumonia", "Tuberculosis", "Other diseases", "No finding"]
    for idx, name in enumerate(class_names):
        print(f"{name}: {label_counts[idx]} samples")

    # Show a few samples
    for i in range(3):
        img, label = dataset[i]
        print(f"Sample {i}: label={label} ({class_names[label]})")

    # Check for PKSampler compatibility
    k = 8  # Set your intended k
    for idx, name in enumerate(class_names):
        if label_counts[idx] < k:
            print(f"Warning: Class '{name}' has fewer than {k} samples ({label_counts[idx]} found).")