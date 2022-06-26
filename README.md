# Deforestation Detection with Fully Convolutional Networks in the Amazon Forest from Landsat-8 and Sentinel-2 Images

Implementation and pretrained models. For details, see * [Paper](https://www.mdpi.com/2072-4292/13/24/5084)

## Description
 This paper comprehensively explores state-of-the-art fully convolutional networks such as U-Net, ResU-Net, SegNet, FC-DenseNet, and two DeepLabv3+ variants on monitoring deforestation in the Brazilian Amazon. The networks’ performance is evaluated experimentally in terms of Precision, Recall, F1-score, and computational load using satellite images with different spatial and spectral resolution: Landsat-8 and Sentinel-2. We also include the results of an unprecedented auditing process performed by senior specialists to visually evaluate each deforestation polygon derived from the network with the highest accuracy results for both satellites. This assessment allowed estimation of the accuracy of these networks simulating a process “in nature” and faithful to the PRODES methodology. We conclude that the high resolution of Sentinel-2 images improves the segmentation of deforestation polygons both quantitatively (in terms of F1-score) and qualitatively. Moreover, the study also points to the potential of the operational use of Deep Learning (DL) mapping as products to be consumed in PRODES

## Training

### Dataset

Location of the Campo Verde dataset in the state of Mato Grosso, Brazil. It comprises 513 fields divided into training (black fields) and validation (gray fields) sets.

<p align="center">
  <img 
    width="800"
    height="800"
    src = study_area_CV.png
  >
</p>

### Dependencies

Please install PyTorch and download the Campo Verde dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the cl_cv_config file. For training please run:
```
python cl_cv.py
```

### SimSiam architecture.

<p align="center">
  <img 
    width="800"
    height="500"
    src = simsiam_last.png
  >
</p>