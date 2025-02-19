# DualAD: Dual Adversarial Network for Image Anomaly Detection

PyTorch implementation of DualAD.

<center><img src="https://img.ziuch.top/i/2024/05/13/12toyau.webp"></center>

## Abstract

>Anomaly Detection, also known as outlier detection, is critical in domains like network security, intrusion detection, and fraud detection. One popular approach to anomaly detection is using autoencoders, which are trained to reconstruct input by minimizing reconstruction error with the neural network. However, these methods usually suffer from the trade-off between normal reconstruction fidelity and abnormal reconstruction distinguishability, which damages the performance. In this paper, we find that the above trade-off can be better mitigated by imposing constraints on the latent space of images. To this end, we propose a new Dual Adversarial Network(DualAD) that consists of a Feature Constraint(FC) module and a reconstruction module. The method incorporates the FC module during the reconstruction training process to impose constraints on the latent space of images, thereby yielding feature representations more conducive to anomaly detection. Additionally, we employ dual adversarial learning to model the distribution of normal data. On the one hand, adversarial learning was implemented during the reconstruction process to obtain higher-quality reconstruction samples, thereby preventing the effects of blurred image reconstructions on model performance. On the other hand, we utilize adversarial training of the FC module and the reconstruction module to achieve superior feature representation, making anomalies more distinguishable at the feature level. During the inference phase, we perform anomaly detection simultaneously in the pixel and latent spaces to identify abnormal patterns more comprehensively. Experiments on three data sets CIFAR10, MNIST, and FashionMNIST demonstrate the validity of our work. Results show that constraints on the latent space and adversarial learning can improve detection performance.

# Datasets

The method is evaluated on:

MNIST dataset：http://yann.lecun.com/exdb/mnist/

FashionMNIST dataset：[fashion-mnist/data/fashion at master · zalandoresearch/fashion-mnist (github.com)](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

CIFAR10 dataset：[CIFAR-10 and CIFAR-100 datasets (toronto.edu)](http://www.cs.toronto.edu/~kriz/cifar.html)
