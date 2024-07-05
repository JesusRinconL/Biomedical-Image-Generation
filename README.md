# Synthesised-Biomedical-Image-Generation
## Introduction
Welcome to the Biomedical Image Generation repository. This repository contains the comprehensive codebase and requirements necessary for generating synthetic biomedical images with pix2pix. It provides detailed explanations for each component, covering the entire process from initial segmentation to the generation of reference images, and culminating in the final model training.

To optimize performance and reduce execution time, it is recommended to run this code on a GPU-enabled device. The codebase is highly adaptable, allowing for modifications to suit various requirements. In particular, the segmentation tool is designed to be customizable, ensuring specificity for different biomedical imaging scenarios.

## Part 0: Enviroment Set Up, Requirements and Data Selection
First of all, it is highly recommended to have a GPU-enabled device, otherwise the execution time is going to be prolonged. The following code checks if you have a GPU operative in your environment:

```python
# Check if CUDA is available
if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. PyTorch is using the CPU.")
```
## Part 1: Image Segmentation
### This part is vinculated to "segmentation.py" file code
Image segmentation is the process of dividing an image into segments or regions of interest (ROIs) that are considered significant. In the context of biomedical imaging, segmentation often involves identifying different cells, their structures, or even their content. This is crucial for measuring morphological characteristics, evaluating the content of structural elements, tracking movements, and assessing treatment effects.

Below is an example of image segmentation, showing how the image is divided into distinct regions:
<div align="center">
    <img src="web/segmentation.png" alt="Image Segmentation Example">
</div>

## Part 2: Image Reference Creation
### There are many file codes to obtain any of the image reference versions created in this work
Once the images are segmented, the detected regions are associated with different categories. This segmented image provides much information and intrinsically describes a pattern unique to each image. Using the segmented image, we define a characteristic pattern that serves as the input for the synthesis model. These patterns can be modified by altering conditions such as image size, number of labels, number of layers, or cell boundary thickness.

Below is an example of the image reference creation process, illustrating the visual changes under different conditions:
<div align="center">
    <img src="web/mask.png" alt="Image Reference Creation Example">
</div>

## Part 3: Image Generation
### Description of the Image Generation Process and Results
Through the defined patterns, datasets are generated to train the deep neural network model. For this work, the pix2pix GAN has been selected due to its effectiveness in generating high-quality images from sketches. The generated synthetic images can be evaluated using objective image evaluation metrics, comparing them with real images to ensure accuracy and realism.

Below is an example of the image generation process, showing the results of synthetic image creation:
<div align="center">
    <img src="web/generation.png" alt="Image Generation Process Example">
</div>
