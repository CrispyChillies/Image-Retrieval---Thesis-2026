# X-MIR: EXplainable Medical Image Retrieval

Code accompanying the WACV '22 paper "X-MIR: EXplainable Medical Image Retrieval"

This code was developed on Ubuntu 18.04 and has been confirmed to work with Pytorch 1.9.0. Please note that newer versions of Pytorch may introduce breaking changes. Other dependencies (such as scikit-learn) may also be required.

## Embedding model for medical image retrieval

This repository trains an embedding model using deep metric learning techniques (e.g. triplet loss), which can then be used for medical image retrieval on chest X-ray or skin lesion images. The main application is to aid clinicians in diagnosing different cases (e.g. COVID or melanoma). This repository makes use of several publicly available resources:

### Datasets
[COVIDx chest X-ray dataset](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md)

Please see the instructions on the Github repo on how to download and prepare the data. Note that this dataset is not static but constantly changes due to new COVID images being added, so your model results may be slightly different than ours. The set of training and testing images we used can be found in `train_split.txt` and `test_COVIDx4.txt`, respectively.

[ISIC 2017 lesion classification dataset](https://challenge.isic-archive.com/data#2017)

We use data from the ISIC 2017 skin lesion classification challenge (please see [this website](https://challenge.isic-archive.com/landing/2017) for more information). The set of training and testing images we used can be found in `ISIC-2017_Training_Part3_GroundTruth.csv` and `ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv`, respectively.

### Pretrained chest X-ray classification model
[CheXNet](https://github.com/arnoweng/CheXNet)

The pretrained Pytorch model weights are provided as `model.pt` in this repository. Please see the Github repo for more information about how this model was trained. When working with chest X-rays, we initialize our similarity models with these pretrained weights.

### Similarity learning using triplet loss
[Torchvision Reference](https://github.com/pytorch/vision/tree/master/references/similarity)

We made use of the above reference example when developing our own deep metric learning training and testing code.

### Training
To train a model, run `python train.py`. By default, trained models will be saved to `./checkpoints`. Please run `python train.py --h` to find out more about the possible command-line options.

For example, to train a baseline chest X-ray similarity model:

```python
 python train.py --dataset-dir '/data/brian.hu/COVID/data' --resume model.pt
 ```

 To train a skin lesion similarity model with an additional 256-dimension embedding layer:

 ```python
 python train.py --dataset 'isic' --dataset-dir '/data/brian.hu/isic/ISIC-2017_Training_Data' --train-image-list 'ISIC-2017_Training_Part3_GroundTruth.csv' --test-image-list 'ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv' --embedding-dim 256
 ```

 **Note**: We also provide experimental functionality to train an "anomaly" version of the model (using the `--anomaly` flag), where the anomaly class (e.g. COVID or melanoma) is not used during training. See the `./anomaly` directory for more information.

### Evaluation
To test a model, run `python test.py`. You must pass an appropriate model path for loading a trained model using the `--resume` flag. **Please make sure the embedding dimension matches that of the trained model using the `--embedding-dim` flag.** By default, model results are saved in the `./results` directory. Please run `python test.py --h` to see all possible command-line options.

For example, to evaluate the COVID chest X-ray model trained above:

```python
python test.py --test-dataset-dir '/data/brian.hu/COVID/data/test' --resume 'covid_densenet121_seed_0_epoch_20_ckpt.pth'
```

 To evaluate the trained ISIC skin lesion model above:

 ```python
python test.py --dataset 'isic' --test-dataset-dir '/data/brian.hu/isic/ISIC-2017_Test_v2_Data' --test-image-list 'ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv' --resume 'isic_densenet121_embed_256_seed_0_epoch_20_ckpt.pth' --embedding-dim 256
```

## Similarity-based saliency maps
This repository also contains implementations of several similarity-based saliency map algorithms, which can be found in `explanations.py`. The code is loosely inspired by the [RISE](https://github.com/eclique/RISE) repository and an earlier version of the [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) repository. The methods below were used in the paper, but several other methods (not described below) are also implemented.

### Implemented Methods
[Explainability for Content-Based Image Retrieval (Dong et al., '19)](http://openaccess.thecvf.com/content_CVPRW_2019/html/Explainable_AI/Dong_Explainability_for_Content-Based_Image_Retrieval_CVPRW_2019_paper.html)  
Black-box saliency method based on occlusion sensitivity

[Visualizing Deep Similarity Networks (Stylianou et al., '19)](https://arxiv.org/abs/1901.00536) / [Visual Explanation for Deep Metric Learning (Zhu et al., '19)](https://arxiv.org/abs/1909.12977)  
White-box saliency method based on pairwise similarity between features in the last convolutional layer

[Learning Similarity Attention (Zheng et al., '19)](https://arxiv.org/abs/1911.07381)  
White-box saliency method based on image triplets: anchor, positive, and negative examples

### Generating Saliency Maps
We also include code for producing saliency maps (including self-similarity saliency maps) which can then be evaluated using different metrics. Please see `compute_saliency.py --h` for more information. **This code currently makes use of Pytorch's [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html), which means it will try to use all GPUs on your machine. To restrict it to only certain GPUs (e.g. GPU 0), you can use `CUDA_VISIBLE_DEVICES=0 python compute_saliency.py`.** Note that only the `sbsm` method has been optimized to make use of multiple GPUs for parallelized saliency map generation (which can be tuned using the `--eval-batch-size` parameter).

### Evaluating Saliency Maps
We also include code for computing the insertion and deletion metrics used to evaluate similarity-based saliency maps (see `evaluate_saliency.py`). At the moment, this requires manually changing the dataset type, paths to the model weights, saliency maps, and test images, and the output filenames in the code. The final results are stored in a `.json` file, where each key is the name of the query image, and the first set of results are the insertion scores and the second set of results are the deletion scores.

Please address all questions to Brian Hu: brian.hu@kitware.com.

### Acknowledgment
This material is based on research sponsored by the Air Force Research Laboratory and DARPA under Cooperative Agreement number N66001-17-2-4028. The U.S. Government is authorized to reproduce and distribute the code for governmental purposes notwithstanding any copyright notation thereon. Distribution Statement "A" (Approved for Public Release, Distribution Unlimited).

### Disclaimer
The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the Air Force Research Laboratory and DARPA or the U.S. Government.