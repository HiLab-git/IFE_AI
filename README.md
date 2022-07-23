# Immunofixation Electrophoresis (IFE) Image Recognition based on Deep Learning
This repository provide source code for IFE image recognition in the following paper:

*H. Hu et al. "Expert-level Immunofixation Electrophoresis (IFE) Image Recognition based on Explainable and Generalizable Deep Learning". Submited to Clinical Chemistry.


# Image and preprocess
The images in this study are from two different systems that have different image styles (see `data/data_a` and `data/data_b`, respectively). We preprocess the images to make them have the same arrangement and size. The following figures shows images before and after preprocessing.

![image](./data/data_a/20200824_1012358442.jpg) ![image_process](./data/data_a_prcess/20200824_1012358442.jpg)

To play with a demo for image preprocessing, run the following command:

```
python preprocess.py
```
