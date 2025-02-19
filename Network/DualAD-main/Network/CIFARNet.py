import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

class AE_CIFAR(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    In_shape = kwargs["input_shape"]
    self.encoder_hidden_layer = nn.Linear(in_features=In_shape, out_features=1024)
    self.encoder_middle_1 = nn.Linear(in_features=1024,out_features=512)
    self.encoder_middle = nn.Linear(in_features=512,out_features=256)
    self.encoder_output_layer = nn.Linear(in_features=256,out_features=256)
    self.decoder_hidden_layer = nn.Linear(in_features=256,out_features=256)
    self.decoder_middle1 = nn.Linear(in_features=256,out_features=512)
    self.decoder_middle = nn.Linear(in_features=512,out_features=1024)
    self.decoder_output_layer = nn.Linear(in_features=1024,out_features=In_shape)

  def forward(self,features):
    activation = F.leaky_relu(self.encoder_middle(F.leaky_relu(self.encoder_middle_1(F.leaky_relu(self.encoder_hidden_layer(features))))))
    code = F.leaky_relu(self.encoder_output_layer(activation)) 
    activation = F.leaky_relu(self.decoder_middle(F.leaky_relu(self.decoder_middle1(F.leaky_relu(self.decoder_hidden_layer(code))))))
    reconstructed = F.leaky_relu(self.decoder_output_layer(activation))
    return reconstructed,code




class AE_CIFAR_Encoder(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    In_shape = kwargs["input_shape"]
    # 定义编码器部分神经网络结构
    self.encoder_hidden_layer = nn.Linear(in_features=In_shape, out_features=1024)
    self.encoder_middle_1 = nn.Linear(in_features=1024,out_features=512)
    self.encoder_middle = nn.Linear(in_features=512,out_features=256)
    self.encoder_output_layer = nn.Linear(in_features=256,out_features=256)


  # 定义前向传播函数，输入 features 是输入数据的特征向量
  def forward(self, features):
    # 实现编码器的前向计算过程
    activation = F.leaky_relu(self.encoder_middle(F.leaky_relu(self.encoder_middle_1(F.leaky_relu(self.encoder_hidden_layer(features))))))
    code = F.leaky_relu(self.encoder_output_layer(activation))
    return code





class Discriminator(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    In_shape = kwargs["input_shape"]
#     self.model = nn.Sequential(
#       nn.Linear(in_features=In_shape, out_features=1024),
#       nn.ReLU(),
#       nn.Linear(in_features=1024, out_features=512),
#       nn.ReLU(),
#       nn.Linear(in_features=512, out_features=1),
#       nn.Sigmoid()
#     )
    self.l1 = nn.Linear(in_features=In_shape, out_features=1024)
    self.l2 = nn.Linear(in_features=1024, out_features=512)
    self.l3 = nn.Linear(in_features=512, out_features=256)
    self.l4 = nn.Linear(in_features=256, out_features=128)
    self.l5 = nn.Linear(in_features=128, out_features=1)

  def forward(self, x):
    x_logits = F.leaky_relu(self.l4(F.leaky_relu(self.l3(F.leaky_relu(self.l2(F.leaky_relu(self.l1(x))))))))
    x_disc = torch.sigmoid(self.l5(x_logits))
#     x_disc = self.model(x)
#     for i, layer in enumerate(self.model):
#         x = layer(x)
#         if i == 3:  # after the second ReLU (index starts at 0)
#             break
    return x_disc, x_logits



class GAN(nn.Module):
  def __init__(self, autoencoder, discriminator):
    super(GAN, self).__init__()
    self.generator = autoencoder
    self.discriminator = discriminator

  def forward(self, x):
    recon_x, _ = self.generator(x)
    disc_x, logits_x = self.discriminator(recon_x)
    return recon_x, disc_x, logits_x

