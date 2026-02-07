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
        
        self.label_columns = [
            "Aortic enlargement", "Cardiomegaly", 
            "Pleural effusion", "Pleural thickening", 
            "Lung Opacity", "No finding"
        ]
        
        df = pd.read_csv(csv_file)
        # Normalize column name: "Other disease" -> "Other diseases"
        if "Other disease" in df.columns and "Other diseases" not in df.columns:
            df = df.rename(columns={"Other disease": "Other diseases"})
        
        if "rad_id" in df.columns:
            self.data = df.groupby("image_id")[self.label_columns].max().reset_index()
        else:
            self.data = df[["image_id"] + self.label_columns].copy()
        
        self.image_ids = self.data["image_id"].tolist()
        self.labels = self.data[self.label_columns].values

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


# ---- ConceptCLIP-compatible VinDR Dataset with concept-rich text generation ----
# Separates disease labels from visual concepts for IT-Align + RC-Align training

# Medical concept descriptions (UMLS-enriched) for concept-rich text generation
CONCEPT_DESCRIPTIONS = {
    "Aortic enlargement": "aortic enlargement with widened mediastinum and dilated aortic contour",
    "Atelectasis": "atelectasis with lung volume loss and collapsed alveolar tissue",
    "Calcification": "calcification with calcified deposits visible as dense opacities",
    "Cardiomegaly": "cardiomegaly with enlarged cardiac silhouette exceeding normal cardiothoracic ratio",
    "Clavicle fracture": "clavicle fracture with disrupted cortical bone continuity",
    "Consolidation": "consolidation with airspace opacification replacing normal lung aeration",
    "Edema": "pulmonary edema with bilateral perihilar haziness and interstitial fluid",
    "Emphysema": "emphysema with hyperinflated lungs and flattened diaphragm",
    "Enlarged PA": "enlarged pulmonary artery suggesting pulmonary hypertension",
    "ILD": "interstitial lung disease with reticular or ground-glass opacities",
    "Infiltration": "pulmonary infiltration with ill-defined opacity in lung parenchyma",
    "Lung Opacity": "lung opacity with abnormal density in the pulmonary field",
    "Lung cavity": "lung cavity with air-filled space surrounded by consolidation or wall",
    "Lung cyst": "lung cyst with thin-walled air-filled space in the lung parenchyma",
    "Mediastinal shift": "mediastinal shift with displacement of central structures",
    "Nodule/Mass": "pulmonary nodule or mass with focal rounded density in the lung",
    "Pleural effusion": "pleural effusion with fluid accumulation in the pleural space",
    "Pleural thickening": "pleural thickening with increased density along the pleural surface",
    "Pneumothorax": "pneumothorax with visible visceral pleural line and absent lung markings",
    "Pulmonary fibrosis": "pulmonary fibrosis with reticular opacities and honeycombing pattern",
    "Rib fracture": "rib fracture with cortical disruption or callus formation",
    "Other lesion": "other lesion with abnormal radiographic finding",
}

DISEASE_DESCRIPTIONS = {
    "COPD": "chronic obstructive pulmonary disease",
    "Lung tumor": "lung tumor or pulmonary malignancy",
    "Pneumonia": "pneumonia with infectious consolidation",
    "Tuberculosis": "tuberculosis with characteristic upper lobe involvement",
    "Other diseases": "other thoracic disease",
    "No finding": "normal chest radiograph without significant pathology",
}


class VINDRConceptCLIPDataSet(Dataset):
    """VinDR dataset for ConceptCLIP fine-tuning.
    
    Returns PIL images (not tensor-transformed) for processing by ConceptCLIP's
    AutoProcessor, along with concept-rich text descriptions, concept labels,
    and disease labels. Separates 22 visual concepts from 6 disease labels.
    """
    
    # 22 visual concepts (radiographic findings)
    CONCEPT_COLUMNS = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
        "Enlarged PA", "ILD", "Infiltration", "Lung Opacity",
        "Lung cavity", "Lung cyst", "Mediastinal shift", "Nodule/Mass",
        "Pleural effusion", "Pleural thickening", "Pneumothorax",
        "Pulmonary fibrosis", "Rib fracture", "Other lesion",
    ]
    
    # 6 disease labels (clinical diagnoses)
    DISEASE_COLUMNS = [
        "COPD", "Lung tumor", "Pneumonia", "Tuberculosis",
        "Other diseases", "No finding",
    ]
    
    ALL_COLUMNS = CONCEPT_COLUMNS + DISEASE_COLUMNS  # 28 total
    
    def __init__(self, data_dir, csv_file, transform=None, return_pil=True):
        """
        Args:
            data_dir: Path to image directory (.png).
            csv_file: Path to CSV file with image_id and label columns.
            transform: Optional image transforms (used when return_pil=False).
            return_pil: If True, return raw PIL images (for ConceptCLIP processor).
                        If False, apply transform and return tensors.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.return_pil = return_pil
        
        df = pd.read_csv(csv_file)
        
        # Normalize column name: test CSV has "Other disease" (singular)
        if "Other disease" in df.columns and "Other diseases" not in df.columns:
            df = df.rename(columns={"Other disease": "Other diseases"})
        
        # Aggregate multi-annotator labels (train has rad_id, test does not)
        if "rad_id" in df.columns:
            self.data = df.groupby("image_id")[self.ALL_COLUMNS].max().reset_index()
        else:
            self.data = df[["image_id"] + self.ALL_COLUMNS].copy()
        
        self.image_ids = self.data["image_id"].tolist()
        self.concept_labels = self.data[self.CONCEPT_COLUMNS].values  # (N, 22)
        self.disease_labels = self.data[self.DISEASE_COLUMNS].values  # (N, 6)
        self.all_labels = self.data[self.ALL_COLUMNS].values          # (N, 28)
        
        # For compatibility with PKSampler/train.py (use all labels)
        self.labels = self.all_labels
    
    def build_text(self, concept_vec, disease_vec):
        """Generate concept-rich text description from label vectors.
        
        Format:  "A chest X-ray showing {disease(s)} with findings of
                  {concept1_description}, {concept2_description}, ..."
                  
        For normal images: "A normal chest X-ray without significant pathological findings."
        
        Args:
            concept_vec: numpy array of shape (22,) with 0/1 values
            disease_vec: numpy array of shape (6,) with 0/1 values
        
        Returns:
            text: concept-rich text description
            concept_names: list of active concept names (for RC-Align)
        """
        active_concepts = [
            self.CONCEPT_COLUMNS[i] for i, v in enumerate(concept_vec) if v == 1
        ]
        active_diseases = [
            self.DISEASE_COLUMNS[i] for i, v in enumerate(disease_vec) if v == 1
        ]
        
        # Check if it's a normal/no-finding image
        is_normal = ("No finding" in active_diseases) and len(active_concepts) == 0
        
        if is_normal:
            text = "A normal chest X-ray without significant pathological findings."
            concept_names = []
        else:
            # Build disease part
            if active_diseases and "No finding" not in active_diseases:
                disease_strs = [DISEASE_DESCRIPTIONS.get(d, d) for d in active_diseases]
                disease_part = ", ".join(disease_strs)
            elif active_diseases:
                # Has "No finding" but also has concepts => treat as findings-only
                disease_part = "unspecified condition"
            else:
                disease_part = "unspecified condition"
            
            # Build concept part with enriched descriptions
            if active_concepts:
                concept_strs = [CONCEPT_DESCRIPTIONS.get(c, c) for c in active_concepts]
                concept_part = ", ".join(concept_strs)
                text = (f"A chest X-ray showing {disease_part} "
                        f"with findings of {concept_part}.")
            else:
                text = f"A chest X-ray showing {disease_part}."
            
            concept_names = active_concepts
        
        return text, concept_names
    
    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.data_dir, f"{img_id}.png")
        
        image = Image.open(img_path).convert('RGB')
        
        concept_vec = self.concept_labels[index]
        disease_vec = self.disease_labels[index]
        all_labels = self.all_labels[index]
        
        # Generate concept-rich text
        text, concept_names = self.build_text(concept_vec, disease_vec)
        
        if not self.return_pil and self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,                                               # PIL Image or Tensor
            'text': text,                                                  # concept-rich description
            'concept_names': concept_names,                                # list of active concept names
            'concept_labels': torch.tensor(concept_vec, dtype=torch.float32),  # (22,)
            'disease_labels': torch.tensor(disease_vec, dtype=torch.float32),  # (6,)
            'all_labels': torch.tensor(all_labels, dtype=torch.float32),       # (28,)
        }
    
    def __len__(self):
        return len(self.image_ids)

# if __name__ == "__main__":
#     dataset = VINDRDataSet(data_dir='', csv_file='/home/aaronpham5504/Coding/Image-Retrieval---Thesis-2026/vindr/image_labels_train.csv')
#     img, target = dataset[0]

#     print(f"Image shape: {img.size}")
#     print(f"Target vector: {target}")
#     print(f"Labels mapping: {dataset.target_columns}")