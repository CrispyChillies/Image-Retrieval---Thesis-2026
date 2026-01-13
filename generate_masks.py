import os
import cv2
import numpy as np
from tqdm import tqdm
from segmentation import segment_and_mask

# Đường dẫn thư mục ảnh gốc và nơi lưu mask
DATA_ROOT = './rsna/data'  # Thay đổi nếu cần
MASK_ROOT = './rsna-png/mask'  # Thư mục lưu mask

for split in ['train', 'test']:
    img_dir = os.path.join(DATA_ROOT, split)
    mask_dir = os.path.join(MASK_ROOT, split)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Lấy danh sách file ảnh (jpg, png)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'Processing {split}: {len(img_files)} images')
    for img_file in tqdm(img_files):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Warning: Cannot read {img_path}')
            continue
        # Sinh mask bằng segment_and_mask
        mask = segment_and_mask(img)
        # Lấy mask nhị phân cuối cùng (ensemble_mask_post_HF_EI)
        # mask đã là nhị phân 0/1, chuyển sang 0/255 để lưu
        mask_bin = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin[mask_bin > 0] = 255
        # Tên file mask: <tên ảnh không đuôi>_xslor.png
        base_name = os.path.splitext(img_file)[0]
        mask_name = base_name + '_xslor.png'
        mask_path = os.path.join(mask_dir, mask_name)
        cv2.imwrite(mask_path, mask_bin)
