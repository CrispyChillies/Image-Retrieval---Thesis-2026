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
            "negative": 1,
            "positive": 2,
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
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        # Segmentation disabled for training
        # # Convert PIL image to numpy array (BGR for OpenCV)
        # image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # # Apply lung segmentation
        # masked_np = segment_and_mask(image_np)
        # # Convert back to PIL Image (RGB)
        # masked_image = Image.fromarray(cv2.cvtColor(masked_np, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)

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


# Test main function for ChestXrayDataSet
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage: update these paths as needed
    data_dir = "./samples"  # directory containing images
    image_list_file = "./train_split.txt"  # file listing images and labels
    dataset = ChestXrayDataSet(data_dir, image_list_file, use_covid=True, mask_dir=None, transform=None)
    print(f"Total images in dataset: {len(dataset)}")
    # Load a sample image
    img, label = dataset[0]
    print(f"Sample label: {label}")
    # Show the image
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()
