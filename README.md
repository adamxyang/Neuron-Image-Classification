# Neuron-Image-Classification
[![License](https://img.shields.io/badge/license-GPLv3-red)](https://github.com/Iron4dam/Neuron-Image-Classification/blob/master/LICENSE) 
[![Python 3.7](https://img.shields.io/badge/python-3.7-yellow.svg)](https://www.python.org/) 
[![PyTorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-blue)](https://pytorch.org) 
[![Torchvision 0.4.0](https://img.shields.io/badge/torchvision-0.4.0-orange)](https://pytorch.org)

# Background
In this work, we aim to build an end-to-end pipeline to estimate treatment effects of Amyloid-β compound in neuronal cells. The pipeline can be further applied to biomedical research to efficiently identify new compounds that protect cells against the treatment of Amyloid-β with potential to discover new drugs for the Alzheimer’s disease. This is a binary classification task where we aim to classify whether neurons are treated with Amyloid-β or not, and we make inference on whether candidate compounds can bind to Amyloid-β and thus protect neurons.

Previous work by [William Stone](https://github.com/wfbstone/Neuron-Image-Classification) has shown the potential of a CNN-based classifiers on this task. He implemented a VGG19 model and achieved 97.2% test accuracy. His training and testing neuron images were taken after staining with the Cy5 dye, a biomarker for the MAP2 gene found in the Neuronal Cytoskeleton. We aim to extend his work to make the model learn from four different stains. Specifically, we stained for MAP2 (neuronal cytoskeleton marker), PSD-95 (post-synaptic protein marker), Synaptophysin (pre-synaptic protein marker) and Hoescht 33342 (nucleic marker). Cells are then imaged at 40x and 20x using the high-content IN Cell Analyzer 6000 imaging system. The images are taken at the internal 48 wells per 36 well plate (the outer wells are lost to edge effects) and obtain 30 fields of view per well for each of the four channels. Cells are pre-treated for one hour with one candidate compound in doses of either 10, 3 or 1 μM2 before the addition of 30 μM of Aβ or a vehicle control. The plate configuration is shown below.
<p align="center">
  <img src="/figures/plate.png" width="600" height='200' title="plate">
</p>
Example images of the four stains:
<p align="center">
  <img src="/figures/4_stains.png" width="1000" height='250' title="stains">
</p>
We will train models using images from 33 out of 36 plates and test the model on the rest three plates. To train the classifier, we only use 0 and 30 μM of Aβ with 0 μM candidate compound. Then we apply the model to test whether candidate compounds have any effect. 

# Models
We first tried to stack four ResNet34 together by concatenating their feature vectors, as shown schematically below. We name this model S4-ResNet34. Due to the high resolution of images (2048x2048) and the GPU memory (RTX 2080 Ti), the deepest convolution based model we can apply is the ResNet34, with mixed-precision training. The reason for such an architecture is that each stain channel has quite distinct structures and they might require different convolutional kernels for feature extraction. Moreover, the four separate convolutional channels also serve as an ensemble method which we
believe can improve the model’s generalisation ability. This model achieved 100.00% accuracy on the test set. 
<p align="center">
  <img src="/figures/Stacked_ResNet.jpg" width="1000" height='500' title="stacked_resnet">
</p>

One major downside of the above model is that it ignores possible mutual spatial information across different stains. Biologists often compare neurons at particular locations using their images from all stains. As different stains highlight different parts of neurons, they can combine the information and make a thorough conclusion. However, our S4-ResNet34 will not be able to do this because it compresses the spatial dimension in each stain channel through pooling before combining the channels. Therefore, we decided to try to stack the channels first, which make training samples RGBY-like images. A schematic diagram of the model is shown below, also with ResNet34 as backbone. This model achieved 99.17% accuracy on the test set. 
<p align="center">
  <img src="/figures/stacked.jpg" width="500" height='250' title="stacked">
</p>

We also experimented other ways to learn from four channels, for example adding all pixel values to combine four channels into one channel, or adding weighted pixels from the fourth channel into the other channels to form a normal RGB image. However, none of the other methods achieved as high accuracy as the above models.

# Inference
We then applied the model to test on wells with various doses of compound and Aβ. Below is an example screening of the plate with compound Danazol:
<p align="center">
  <img src="/figures/danazol.png" width="600" height='200' title="danazol">
</p>
Our results align with biologists' statistical analysis using [CellProfiler](https://cellprofiler.org). But once our model is trained, the inference time was 1000 time faster than the traditional method using CellProfiler. We also implemented GradCAM to visualise pixel-wise importance:
<p align="center">
  <img src="/figures/gradcam.png" width="600" height='600' title="gradcam">
</p>
