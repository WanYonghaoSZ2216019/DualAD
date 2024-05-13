import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms
from .Utils import get_target_label_idx,global_contrast_normalization,OneClass
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random


# 定义函数MNIST_loader，用于加载MNIST数据集并返回train_loader、test_loader以及对应的标签labels
def MNIST_loader(train_batch, test_batch, Class):
  # 将数据转换为PyTorch张量
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  # 当 train=False 时，torchvision.datasets.MNIST 函数将加载测试集数据，返回一个包含测试集数据和标签的 torch.utils.data.Dataset 对象。而当 train=True 时，函数将加载训练集数据，并返回一个包含训练集数据和标签的 torch.utils.data.Dataset 对象
  train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
  test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)


  # 利用OneClass函数得到训练数据、测试数据以及测试数据的标签，Digits是训练数据，new_test, labels分别是测试数据和测试数据的标签
  Digits, new_test, labels = OneClass(train_dataset, test_dataset, Class)
  train_loader = torch.utils.data.DataLoader(Digits, batch_size=train_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test, batch_size=test_batch, shuffle=False, num_workers=0)
  # 返回两个数据集加载器，以及测试集的标签
  return train_loader, test_loader, labels

# 与MNIST_loader同理
def FMNIST_loader(train_batch,test_batch,Class):
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets",train=True,transform=transform,download=True)
  test_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets", train=False, transform=transform, download=True)


  Digits,new_test,labels = OneClass(train_dataset,test_dataset,Class)
  train_loader = torch.utils.data.DataLoader(Digits,batch_size=train_batch,shuffle=True,num_workers=0,pin_memory=True,drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test,batch_size=test_batch,shuffle=False,num_workers=0)
  return train_loader,test_loader, labels

def CIFAR_loader(train_batch,test_batch,Class):
  min_max = [(-28.94083453598571, 13.802961825439636),
              (-6.681770233365245, 9.158067708230273),
              (-34.924463588638204, 14.419298165027628),
              (-10.599172931391799, 11.093187820377565),
              (-11.945022995801637, 10.628045447867583),
              (-9.691969487694928, 8.948326776180823),
              (-9.174940012342555, 13.847014686472365),
              (-6.876682005899029, 12.282371383343161),
              (-15.603507135507172, 15.2464923804279),
              (-6.132882973622672, 8.046098172351265)]
  transform = transforms.Compose(
      [transforms.ToTensor(),transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l2'))])

  train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)


  test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  Digits,new_test,labels = OneClass(train_dataset,test_dataset,Class)
  train_loader = torch.utils.data.DataLoader(Digits,batch_size=train_batch,shuffle=True,num_workers=0,pin_memory=True,drop_last = True)
  test_loader = torch.utils.data.DataLoader(new_test,batch_size=test_batch,shuffle=False,num_workers=0)
  return train_loader,test_loader,labels




def random_cifar():
    # 定义数据预处理
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 为每个类创建一个样本索引列表
    class_indices = {i: [] for i in range(10)}
    for idx, (_, class_label) in enumerate(train_dataset):
      class_indices[class_label].append(idx)

    # 随机选择每个类中的两个样本并修改目标标签
    num_samples_per_class = 2
    random_indices = []
    modified_targets = []
    for class_label in range(10):
      selected_indices = random.sample(class_indices[class_label], num_samples_per_class)
      random_indices.extend(selected_indices)

      # 将类标签修改为0或1，如果类标签为3，则将其修改为1，否则将其修改为0
      new_targets = [1 if class_label == 3 else -1 for _ in range(num_samples_per_class)]
      modified_targets.extend(new_targets)

    # 从原始数据集中创建一个包含所选样本的子集
    selected_subset = Subset(train_dataset, random_indices)

    # 用新的目标标签替换原始目标标签
    selected_subset = list(selected_subset)  # 将Subset转换为列表
    for i, (data, _) in enumerate(selected_subset):
      selected_subset[i] = (data, modified_targets[i])

    # 创建一个数据加载器
    selected_dataloader = DataLoader(selected_subset, batch_size=20, shuffle=True)

    # 获取一批大小为20的数据
    data, targets = next(iter(selected_dataloader))
    # print("Data shape: ", data.shape)
    # print("Targets: ", targets)
    return data, targets

def random_select(Class=5):
  # 下载并加载MNIST数据集
  transform = transforms.Compose([transforms.ToTensor()])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=30, shuffle=True)

  # 获取一个batch大小为30的样本
  images, labels = iter(trainloader).next()

  # 定义目标类别
  target_class = Class

  # 将目标类的标签设为1，其他类的标签设为-1
  binary_labels = (labels == target_class).float() * 2 - 1

  # 返回样本、设置标签以及原始标签
  return images, binary_labels, labels



def get_balanced_samples(target_class, num_samples=30):
    assert num_samples % 2 == 0, "Please provide an even number for num_samples."

    # 下载和加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    # 为目标类和非目标类初始化计数器和样本列表
    target_class_count, other_class_count = 0, 0
    num_each_class = num_samples // 2
    selected_samples, binary_labels, original_labels = [], [], []

    for image, label in trainloader:
      if label.item() == target_class and target_class_count < num_each_class:
        selected_samples.append(image)
        binary_labels.append(1.0)
        original_labels.append(label.item())
        target_class_count += 1
      elif label.item() != target_class and other_class_count < num_each_class:
        selected_samples.append(image)
        binary_labels.append(-1.0)
        original_labels.append(label.item())
        other_class_count += 1

      if target_class_count == num_each_class and other_class_count == num_each_class:
        break

    # 将列表转换为张量
    selected_samples = torch.cat(selected_samples, dim=0)
    binary_labels = torch.tensor(binary_labels)
    original_labels = torch.tensor(original_labels)

    return selected_samples, binary_labels, original_labels



def get_negative_samples(target_class, num_samples=30):
    # 下载和加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    # 初始化样本计数器和列表
    selected_count = 0
    selected_samples, binary_labels, original_labels = [], [], []

    for image, label in trainloader:
        if label.item() != target_class and selected_count < num_samples:
            selected_samples.append(image)
            binary_labels.append(-1.0)
            original_labels.append(label.item())
            selected_count += 1

        if selected_count == num_samples:
            break

    # 将列表转换为张量
    selected_samples = torch.cat(selected_samples, dim=0)
    binary_labels = torch.tensor(binary_labels)
    original_labels = torch.tensor(original_labels)

    return selected_samples, binary_labels, original_labels




def random_mnist():
    # 定义数据预处理
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 为每个类创建一个样本索引列表
    class_indices = {i: [] for i in range(10)}
    for idx, (_, class_label) in enumerate(train_dataset):
      class_indices[class_label].append(idx)

    # 随机选择每个类中的两个样本并修改目标标签
    num_samples_per_class = 2
    random_indices = []
    modified_targets = []
    for class_label in range(10):
      selected_indices = random.sample(class_indices[class_label], num_samples_per_class)
      random_indices.extend(selected_indices)

      # 将类标签修改为0或1，如果类标签为3，则将其修改为1，否则将其修改为0
      new_targets = [1 if class_label == 3 else -1 for _ in range(num_samples_per_class)]
      modified_targets.extend(new_targets)

    # 从原始数据集中创建一个包含所选样本的子集
    selected_subset = Subset(train_dataset, random_indices)

    # 用新的目标标签替换原始目标标签
    selected_subset = list(selected_subset)  # 将Subset转换为列表
    for i, (data, _) in enumerate(selected_subset):
      selected_subset[i] = (data, modified_targets[i])

    # 创建一个数据加载器
    selected_dataloader = DataLoader(selected_subset, batch_size=len(selected_subset), shuffle=True)

    # 获取一批大小为20的数据
    data, targets = next(iter(selected_dataloader))
    # print("Data shape: ", data.shape)
    # print("Targets: ", targets)
    return data, targets