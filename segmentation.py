import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from scipy import ndimage
from scipy.ndimage import label, generate_binary_structure

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["SM_FRAMEWORK"] = "tf.keras"

import efficientnet.tfkeras as efn
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')


IMAGE_SIZE = (256,256,3)

path_base_model = './models/'
path_base_input = './samples/'

BACKBONE = 'efficientnetb0'

# Khởi tạo các model B0
model1 = sm.Unet(BACKBONE, input_shape=IMAGE_SIZE, classes=1, activation='sigmoid', encoder_weights='imagenet')
model2 = sm.Unet(BACKBONE, input_shape=IMAGE_SIZE, classes=1, activation='sigmoid', encoder_weights='imagenet')
model3 = sm.Unet(BACKBONE, input_shape=IMAGE_SIZE, classes=1, activation='sigmoid', encoder_weights='imagenet')

BACKBONE = 'efficientnetb7'
# Khởi tạo các model B7
model4 = sm.Unet(BACKBONE, input_shape=IMAGE_SIZE, classes=1, activation='sigmoid', encoder_weights='imagenet')
model5 = sm.Unet(BACKBONE, input_shape=IMAGE_SIZE, classes=1, activation='sigmoid', encoder_weights='imagenet')

preprocess_input = sm.get_preprocessing(BACKBONE)

model1.load_weights(path_base_model + 'model1.hdf5')
model2.load_weights(path_base_model + 'model2.hdf5')
model3.load_weights(path_base_model + 'model3.hdf5')
model4.load_weights(path_base_model + 'model4.hdf5')
model5.load_weights(path_base_model + 'model5.hdf5')


def preprocessing_HE(img_):
    
    hist, bins = np.histogram(img_.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img_2 = cdf[img_]
    
    return img_2  
        
def get_binary_mask (mask_, th_ = 0.5):
    mask_[mask_>th_]  = 1
    mask_[mask_<=th_] = 0
    return mask_
    
def ensemble_results (mask1_, mask2_, mask3_, mask4_, mask5_):
    
    mask1_ = get_binary_mask (mask1_)
    mask2_ = get_binary_mask (mask2_)
    mask3_ = get_binary_mask (mask3_)
    mask4_ = get_binary_mask (mask4_)
    mask5_ = get_binary_mask (mask5_)
    
    ensemble_mask = mask1_ + mask2_ + mask3_ + mask4_ + mask5_
    ensemble_mask[ensemble_mask<=2.0] = 0
    ensemble_mask[ensemble_mask> 2.0] = 1
    
    return ensemble_mask

def postprocessing_HoleFilling (mask_):
    
    ensemble_mask_post_temp = ndimage.binary_fill_holes(mask_).astype(int)
     
    return ensemble_mask_post_temp

def get_maximum_index (labeled_array):
    
    ind_nums = []
    for i in range (len(np.unique(labeled_array)) - 1):
        ind_nums.append ([0, i+1])
        
    for i in range (1, len(np.unique(labeled_array))):
        ind_nums[i-1][0] = len(np.where (labeled_array == np.unique(labeled_array)[i])[0])
        
    ind_nums = sorted(ind_nums)
    
    return ind_nums[len(ind_nums)-1][1], ind_nums[len(ind_nums)-2][1]
    
def postprocessing_EliminatingIsolation (ensemble_mask_post_temp):
        
    labeled_array, num_features = label(ensemble_mask_post_temp)
    
    ind_max1, ind_max2 = get_maximum_index (labeled_array)
    
    ensemble_mask_post_temp2 = np.zeros (ensemble_mask_post_temp.shape)
    ensemble_mask_post_temp2[labeled_array == ind_max1] = 1
    ensemble_mask_post_temp2[labeled_array == ind_max2] = 1    
    
    return ensemble_mask_post_temp2.astype(int)

def get_prediction(model_, img_org_):
    
    img_org_resize = cv2.resize(img_org_,(IMAGE_SIZE[0],IMAGE_SIZE[1]),cv2.INTER_AREA)
    img_org_resize_HE = preprocessing_HE (img_org_resize)    
    img_ready = preprocess_input (img_org_resize_HE)

    img_ready = np.expand_dims(img_ready, axis=0) 
    pr_mask = model_.predict(img_ready)
    pr_mask = np.squeeze(pr_mask)
    pr_mask = np.expand_dims(pr_mask, axis=-1)    
    return pr_mask[:,:,0]



def segment_and_mask(img):
    """
    Run ensemble segmentation and apply the final mask to the original image.
    Args:
        img: Input image as a numpy array (H, W, 3), BGR (cv2.imread)
    Returns:
        masked_img: The original image with the ensemble mask (ensemble_mask_post_HF_EI) applied
    """
    pr_mask1 = get_prediction(model1, img)
    pr_mask2 = get_prediction(model2, img)
    pr_mask3 = get_prediction(model3, img)
    pr_mask4 = get_prediction(model4, img)
    pr_mask5 = get_prediction(model5, img)

    ensemble_mask = ensemble_results(pr_mask1, pr_mask2, pr_mask3, pr_mask4, pr_mask5)
    ensemble_mask_post_HF = postprocessing_HoleFilling(ensemble_mask)
    ensemble_mask_post_HF_EI = postprocessing_EliminatingIsolation(ensemble_mask_post_HF)

    # Resize mask to original image size if needed
    mask_resized = cv2.resize(ensemble_mask_post_HF_EI.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Apply mask: keep only lung region, set background to black
    masked_img = cv2.bitwise_and(img, img, mask=mask_resized)
    return masked_img

# def main():
#     import sys
#     # Example: test on all images in path_base_input
#     for path_ in sorted(glob.glob(path_base_input + '*.*')):
#         print('file:', path_.split('/')[-1])
#         img = cv2.imread(path_)
#         masked_img = segment_and_mask(img)
#         # Show original and masked image side by side
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.title('Original')
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.subplot(1, 2, 2)
#         plt.title('Masked (Lung Only)')
#         plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()
#     print("Hello")

# if __name__ == "__main__":
#     main()