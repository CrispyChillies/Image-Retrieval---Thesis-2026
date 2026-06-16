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
from pathlib import Path
from urllib.parse import unquote
NIH_ORIGINAL_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumothorax",
    "Emphysema",
    "Pleural Thickening",
    "Fibrosis",
    "Consolidation",
    "Edema",
    "Pneumonia",
    "Infiltration",
    "Nodule",
    "Mass",
]

NIH_U_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Pneumothorax",
    "Emphysema",
    "Pleural Thickening",
    "Fibrosis",
    "Opacities",
    "Lesion",
]

NIH_U_MAPPING = {
    "Atelectasis": ["Atelectasis"],
    "Cardiomegaly": ["Cardiomegaly"],
    "Effusion": ["Effusion"],
    "Pneumothorax": ["Pneumothorax"],
    "Emphysema": ["Emphysema"],
    "Pleural Thickening": ["Pleural Thickening"],
    "Fibrosis": ["Fibrosis"],
    "Opacities": ["Consolidation", "Edema", "Pneumonia", "Infiltration"],
    "Lesion": ["Nodule", "Mass"],
}

NIH_RETRIEVAL_PATHOLOGIES = NIH_U_LABELS
NIH_NPY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]
NIH_NPY_LABEL_ALIASES = {
    "pleural_thickening": "Pleural Thickening",
    "pleural thickening": "Pleural Thickening",
    "pleuralthickening": "Pleural Thickening",
}
NIH_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NIH_IMAGE_SHARD_PREFIX = "images_"


def _normalize_nih_label(label_name):
    return label_name.strip().replace("_", " ").replace("-", " ").lower()


def _read_nih_image_list(image_list_file=None):
    image_names = []
    if image_list_file:
        manifest_path = Path(image_list_file)
        if manifest_path.is_file():
            if manifest_path.suffix.lower() == ".csv":
                df = pd.read_csv(manifest_path)
                image_col = _find_nih_image_column(df)
                image_names = df[image_col].dropna().astype(str).str.strip().tolist()
            else:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if line:
                            image_names.append(line.split(",")[0].strip())
    return image_names


def _resolve_nih_npy_paths(data_dir=None, image_list_file=None):
    paths = []

    if image_list_file:
        manifest_path = Path(image_list_file)
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    candidate = Path(line.split(",")[0].strip())
                    if not candidate.is_absolute() and data_dir is not None:
                        candidate = Path(data_dir) / candidate
                    if candidate.suffix.lower() == ".npy" and candidate.is_file():
                        paths.append(str(candidate))

    if not paths and data_dir:
        paths = sorted(str(path) for path in Path(data_dir).rglob("*.npy"))

    return paths


def _to_uint8_image(array):
    array = np.asarray(array)

    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)

    array = (array - min_value) / (max_value - min_value)
    array = np.clip(array * 255.0, 0.0, 255.0)
    return array.astype(np.uint8)


def _find_nih_image_column(df):
    for column in ("Image Index", "image_id", "image", "filename", "fname", "path"):
        if column in df.columns:
            return column
    raise ValueError(
        "NIH metadata CSV must contain an image column such as 'Image Index'. "
        f"Found columns: {list(df.columns)}"
    )


def _find_nih_label_column(df):
    for column in ("Finding Labels", "Finding Label", "labels", "label"):
        if column in df.columns:
            return column
    raise ValueError(
        "NIH metadata CSV must contain a label column such as 'Finding Labels'. "
        f"Found columns: {list(df.columns)}"
    )


def _find_nih_metadata_csv(data_dir=None, image_list_file=None, labels_csv_file=None):
    candidates = []
    if labels_csv_file:
        candidates.append(Path(labels_csv_file))
    if image_list_file and Path(image_list_file).suffix.lower() == ".csv":
        candidates.append(Path(image_list_file))
    if data_dir:
        data_path = Path(data_dir)
        candidates.extend(
            [
                data_path / "Data_Entry_2017.csv",
                data_path.parent / "Data_Entry_2017.csv",
            ]
        )
    candidates.extend([Path("nih") / "Data_Entry_2017.csv", Path("Data_Entry_2017.csv")])

    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find NIH label metadata CSV. Pass --nih-labels-csv pointing to "
        f"Data_Entry_2017.csv or another CSV with image and label columns. Searched: {searched}"
    )


def _build_nih_path_index(data_dir):
    path_index = {}
    if not data_dir:
        return path_index

    data_path = Path(data_dir)
    if not data_path.exists():
        return path_index

    image_roots = [
        shard_dir / "images"
        for shard_dir in sorted(data_path.glob(f"{NIH_IMAGE_SHARD_PREFIX}*"))
        if shard_dir.is_dir() and (shard_dir / "images").is_dir()
    ]
    if data_path.name == "images":
        image_roots.append(data_path)
    if not image_roots:
        direct_images_dir = data_path / "images"
        if direct_images_dir.is_dir():
            image_roots.append(direct_images_dir)

    search_roots = image_roots or [data_path]
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in NIH_IMAGE_EXTENSIONS:
                path_index.setdefault(path.name, path)
    return path_index


def _describe_nih_image_layout(data_dir):
    if not data_dir:
        return "No data_dir was provided."
    data_path = Path(data_dir)
    shard_example = data_path / "images_001" / "images"
    return (
        "Expected NIH images under the dataset root using the original layout, "
        f"for example '{shard_example}', or under a direct 'images' folder."
    )


class NIHChestXrayRetrievalDataSet(Dataset):
    """NIH chest X-ray retrieval dataset backed by PNG/JPEG images.

    Labels are read from the official NIH metadata CSV and collapsed from the
    original NIH findings into the unified labels in ``NIH_U_LABELS``.
    """

    def __init__(
        self,
        data_dir=None,
        image_list_file=None,
        labels_csv_file=None,
        transform=None,
        pathology_names=None,
        label_mapping=None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        npy_paths = _resolve_nih_npy_paths(data_dir=data_dir, image_list_file=image_list_file)
        if npy_paths:
            self.image_names = npy_paths
            self.pathology_names = list(pathology_names or NIH_NPY_LABELS)
            self.pathology_to_index = {
                name: idx for idx, name in enumerate(self.pathology_names)
            }
            self.pathology_aliases = NIH_NPY_LABEL_ALIASES.copy()
            for name in self.pathology_names:
                self.pathology_aliases[self._normalize_npy_label(name)] = name

            self.labels = []
            self.label_sets = []
            for image_path in self.image_names:
                label_names, multi_hot = self._parse_npy_labels_from_path(image_path)
                self.label_sets.append(label_names)
                self.labels.append(multi_hot)
            return

        self.pathology_names = list(pathology_names or NIH_U_LABELS)
        self.label_mapping = label_mapping or NIH_U_MAPPING
        self.pathology_to_index = {name: idx for idx, name in enumerate(self.pathology_names)}
        self.original_to_unified = self._build_original_to_unified(self.label_mapping)

        metadata_path = _find_nih_metadata_csv(
            data_dir=data_dir,
            image_list_file=image_list_file,
            labels_csv_file=labels_csv_file,
        )
        metadata_df = pd.read_csv(metadata_path)
        image_col = _find_nih_image_column(metadata_df)
        label_col = _find_nih_label_column(metadata_df)

        metadata_by_image = {}
        for _, row in metadata_df.iterrows():
            image_name = str(row[image_col]).strip()
            raw_labels = str(row[label_col]).strip()
            if image_name:
                image_key = Path(image_name).name
                if image_key in metadata_by_image:
                    metadata_by_image[image_key] = (
                        f"{metadata_by_image[image_key]}|{raw_labels}"
                    )
                else:
                    metadata_by_image[image_key] = raw_labels

        requested_images = _read_nih_image_list(image_list_file)
        if not requested_images:
            requested_images = sorted(metadata_by_image)
        requested_images = list(dict.fromkeys(requested_images))

        path_index = _build_nih_path_index(data_dir)
        self.image_names = []
        self.labels = []
        self.label_sets = []
        missing_metadata = []
        missing_images = []

        for image_name in requested_images:
            image_path = self._resolve_image_path(image_name, path_index)
            image_key = Path(image_path).name
            raw_labels = metadata_by_image.get(image_key)
            if raw_labels is None:
                missing_metadata.append(image_key)
                continue

            if not Path(image_path).is_file():
                missing_images.append(image_path)
                continue

            label_names, multi_hot = self._encode_labels(raw_labels)
            self.image_names.append(str(image_path))
            self.label_sets.append(label_names)
            self.labels.append(multi_hot)

        if missing_metadata:
            raise ValueError(
                f"Missing NIH metadata for {len(missing_metadata)} images. "
                f"Examples: {missing_metadata[:5]}"
            )
        if missing_images:
            raise FileNotFoundError(
                f"Missing NIH image files for {len(missing_images)} entries. "
                f"Examples: {missing_images[:5]}. {_describe_nih_image_layout(data_dir)}"
            )
        if not self.image_names:
            raise ValueError(
                "No NIH images were loaded. "
                f"Check data_dir, image_list_file, and labels_csv_file. {_describe_nih_image_layout(data_dir)}"
            )

    def _build_original_to_unified(self, label_mapping):
        original_to_unified = {}
        for unified_label, original_labels in label_mapping.items():
            if unified_label not in self.pathology_to_index:
                continue
            for original_label in original_labels:
                original_to_unified[_normalize_nih_label(original_label)] = unified_label
        return original_to_unified

    def _normalize_npy_label(self, label_name):
        return (
            label_name.strip()
            .replace("%20", " ")
            .replace("_", " ")
            .replace("-", " ")
            .lower()
        )

    def _parse_npy_labels_from_path(self, image_path):
        stem = Path(image_path).stem
        prefix = "Chest_X-ray_"
        prefix_index = stem.find(prefix)
        if prefix_index < 0:
            raise ValueError(
                f"Unsupported NIH file name '{Path(image_path).name}'. "
                f"Expected token '{prefix}'."
            )

        stem_without_prefix = stem[prefix_index + len(prefix):]
        try:
            encoded_labels, _ = stem_without_prefix.rsplit("_", 1)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported NIH file name '{Path(image_path).name}'. "
                "Expected labels and numeric identifier separated by the final underscore."
            ) from exc

        raw_label_names = [label.strip() for label in unquote(encoded_labels).split("|")]
        label_names = []
        multi_hot = np.zeros(len(self.pathology_names), dtype=np.float32)
        unknown_labels = []
        for raw_label in raw_label_names:
            normalized_label = self._normalize_npy_label(raw_label)
            canonical_label = self.pathology_aliases.get(normalized_label)
            if canonical_label is None:
                unknown_labels.append(raw_label)
                continue
            label_idx = self.pathology_to_index.get(canonical_label)
            if label_idx is None:
                unknown_labels.append(raw_label)
                continue
            multi_hot[label_idx] = 1.0
            label_names.append(canonical_label)

        if unknown_labels:
            raise ValueError(
                f"Unknown pathologies in '{Path(image_path).name}': {unknown_labels}. "
                f"Known labels: {self.pathology_names}"
            )

        return label_names, multi_hot

    def _resolve_image_path(self, image_name, path_index):
        image_path = Path(image_name)
        if image_path.is_absolute():
            return image_path
        if self.data_dir:
            direct_path = Path(self.data_dir) / image_path
            if direct_path.is_file():
                return direct_path
        return path_index.get(image_path.name, Path(self.data_dir or ".") / image_path)

    def _encode_labels(self, raw_labels):
        label_names = []
        multi_hot = np.zeros(len(self.pathology_names), dtype=np.float32)
        for raw_label in raw_labels.split("|"):
            normalized_label = _normalize_nih_label(raw_label)
            if normalized_label in ("", "no finding", "hernia"):
                continue
            unified_label = self.original_to_unified.get(normalized_label)
            if unified_label is None:
                continue
            multi_hot[self.pathology_to_index[unified_label]] = 1.0
            label_names.append(unified_label)

        return sorted(set(label_names)), multi_hot

    def __getitem__(self, index):
        image_path = self.image_names[index]
        if str(image_path).lower().endswith(".npy"):
            image_array = np.load(image_path)
            image_array = _to_uint8_image(image_array)
            image = Image.fromarray(image_array).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.image_names)


class ISICDataSet(Dataset):
    def __init__(
        self,
        data_dir,
        image_list_file,
        use_melanoma=True,
        mask_dir=None,
        transform=None,
    ):
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
        with open(image_list_file, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for line in reader:
                image_name = line[0] + ".jpg"
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
        image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new("RGB", image.size), mask)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)


class ChestXrayDataSet(Dataset):
    def __init__(
        self, data_dir, image_list_file, use_covid=True, mask_dir=None, transform=None
    ):
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
                        mask_dir, os.path.splitext(image_name)[0] + "_xslor.png"
                    )
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
        image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new("RGB", image.size), mask)
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

        with open(csv_file, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header row: {csv_file}")

            # Normalize headers to avoid BOM/case/whitespace issues.
            normalized_field_map = {
                name.strip().lstrip("\ufeff").lower(): name
                for name in reader.fieldnames
                if name is not None
            }
            fname_key = normalized_field_map.get("fname")
            image_type_key = normalized_field_map.get("image_type")

            if fname_key is None or image_type_key is None:
                raise ValueError(
                    "TBX11k CSV must contain 'fname' and 'image_type' columns. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row in reader:
                fname = row.get(fname_key, "").strip()
                image_type = row.get(image_type_key, "").strip()
                if not fname or not image_type:
                    continue
                # Only use images with valid image_type
                if image_type not in self.type_map:
                    continue
                img_path = os.path.join(data_dir, fname)
                self.image_names.append(img_path)
                self.labels.append(self.type_map[image_type])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert("RGB")
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

        image = Image.open(img_path).convert("RGB")

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
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Clavicle fracture",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Enlarged PA",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Lung cavity",
        "Lung cyst",
        "Mediastinal shift",
        "Nodule/Mass",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis",
        "Rib fracture",
        "Other lesion",
    ]

    # 6 disease labels (clinical diagnoses)
    DISEASE_COLUMNS = [
        "COPD",
        "Lung tumor",
        "Pneumonia",
        "Tuberculosis",
        "Other diseases",
        "No finding",
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
        self.all_labels = self.data[self.ALL_COLUMNS].values  # (N, 28)

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
                text = (
                    f"A chest X-ray showing {disease_part} "
                    f"with findings of {concept_part}."
                )
            else:
                text = f"A chest X-ray showing {disease_part}."

            concept_names = active_concepts

        return text, concept_names

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.data_dir, f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")

        concept_vec = self.concept_labels[index]
        disease_vec = self.disease_labels[index]
        all_labels = self.all_labels[index]

        # Generate concept-rich text
        text, concept_names = self.build_text(concept_vec, disease_vec)

        if not self.return_pil and self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,  # PIL Image or Tensor
            "text": text,  # concept-rich description
            "concept_names": concept_names,  # list of active concept names
            "concept_labels": torch.tensor(concept_vec, dtype=torch.float32),  # (22,)
            "disease_labels": torch.tensor(disease_vec, dtype=torch.float32),  # (6,)
            "all_labels": torch.tensor(all_labels, dtype=torch.float32),  # (28,)
        }

    def __len__(self):
        return len(self.image_ids)


# if __name__ == "__main__":
#     dataset = VINDRDataSet(data_dir='', csv_file='/home/aaronpham5504/Coding/Image-Retrieval---Thesis-2026/vindr/image_labels_train.csv')
#     img, target = dataset[0]

#     print(f"Image shape: {img.size}")
#     print(f"Target vector: {target}")
#     print(f"Labels mapping: {dataset.target_columns}")
