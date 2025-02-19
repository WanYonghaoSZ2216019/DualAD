import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms
from Dataset.Utils import get_target_label_idx,global_contrast_normalization,OneClass
from Dataset.DatasetLoader import MNIST_loader, FMNIST_loader, CIFAR_loader,Speech_loader,PIMA_loader, get_negative_samples, random_cifar, get_balanced_samples, random_select
from Network.CIFARNet import AE_CIFAR, AE_CIFAR_Encoder, Discriminator, GAN
from Network.MNISTNet import AE_MNIST
from Network.PIMANet import AE_PIMA
from Network.SpeechNet import AE_Speech
from Source.GammaTune import tune_gamma
from Source.Tester import DASVDD_test
from Source.Trainer import DASVDD_trainer
import random


def set_seed(seed_value):
    """设置所有需要的随机种子以确保结果的可重复性。"""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

seed = 42  # 你可以选择任何数字作为种子值
set_seed(seed)


def init_network_weights_from_pretraining(net, ae_net):
    """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

    net_dict = net.state_dict()  # 获取Deep SAD网络的状态字典
    ae_net_dict = ae_net.state_dict()  # 获取预训练自编码器的状态字典

    # Filter out decoder network keys
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}  # 除预训练自编码器的解码层的权重，因为Deep SAD网络没有解码层
    # Overwrite values in the existing state_dict
    net_dict.update(ae_net_dict)  # 将剩余的编码器权重添加到Deep SAD网络的状态字典中
    # Load the new state_dict
    net.load_state_dict(net_dict)  # 重新加载状态字典，使网络的权重和预训练自编码器的编码器权重相同
    return net

def update_ae_net_from_net(net, ae_net):
    """Update the autoencoder (ae_net) weights with encoder weights from the Deep SAD network (net)."""


    net_dict = net.state_dict()  # 获取 Deep SAD 网络 net 的状态字典
    ae_net_dict = ae_net.state_dict()  # 获取自编码器网络 ae_net 的状态字典

    # 过滤出编码器层的权重，因为在 ae_net 中只关心编码器部分
    net_dict = {k: v for k, v in net_dict.items() if k in ae_net_dict}

    # 更新 ae_net_dict 的权重
    ae_net_dict.update(net_dict)

    # 重新加载更新后的状态字典 ae_net_dict 到 ae_net
    ae_net.load_state_dict(ae_net_dict)
    return ae_net


# 加载MNIST数据集，并设定训练集batch_size为200，测试集batch_size为1,正类标签为class
train_loader, test_loader, labels = CIFAR_loader(train_batch=64, test_batch=1, Class=8)
# 定义输入图片的形状in_shape为28*28
in_shape = 3*32*32
# 定义隐藏层的编码维度code_size为256
code_size = 256
# 如果有GPU，则将模型放到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 构建一个输入形状为in_shape的MNIST自编码器模型model，并将其放到设备device上
model = AE_CIFAR(input_shape=in_shape).to(device)
discriminator = Discriminator(input_shape=in_shape)
discriminator = discriminator.to(device)
gan = GAN(model, discriminator).to(device)

# 获取该模型的所有参数
params = list(model.parameters())
params1 = list(discriminator.parameters())
# 定义优化器为Adam优化器，并将其应用于所有模型参数
optimizer = torch.optim.Adam(params, lr=1e-5)
optimizer_discriminator = torch.optim.Adam(params1, lr=1e-5)
# 随机生成一个编码中心C，并将其放到设备device上，设置requires_grad=True，表示此张量需要梯度计算
C = torch.randn(code_size, device=device, requires_grad=True)
# 定义编码中心C的优化器为Adagrad优化器，学习率为1，学习率衰减为0.01
update_center = torch.optim.Adagrad([C], lr=1, lr_decay=0.01)
# 定义损失函数为MSE（均方误差）损失函数
criterion = nn.MSELoss()
criterion1 = nn.BCELoss()
# 利用函数tune_gamma计算所需的Gamma值，此处传入的AE_MNIST为上一步定义的MNIST自编码器模型，T为温度参数（默认为10）
Gamma = tune_gamma(AE_CIFAR, in_shape, criterion, train_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), T=10)



# 利用函数DASVDD_trainer对模型进行训练，其中传入的参数分别为：模型model、输入形状in_shape、隐藏层编码维度code_size、编码中心C、训练数据集train_loader
# 、模型优化器optimizer、编码中心更新器update_center、损失函数criterion、Gamma值、设备device、训练轮数num_epochs、K值（默认为0.9）--迭代时的数据分配
DASVDD_trainer(model,DASVDD_test, gan, discriminator, in_shape, code_size, C, train_loader, test_loader, labels, optimizer, optimizer_discriminator, update_center, criterion, criterion1, Gamma, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=300, K=0.9)


net = AE_CIFAR_Encoder(input_shape=in_shape).to(device)
net = init_network_weights_from_pretraining(net, model)
params1 = list(net.parameters())
# 定义优化器为Adam优化器，并将其应用于所有模型参数
optimizer1 = torch.optim.Adam(params1, lr=1e-6)

# data, targets, _ = get_negative_samples(target_class=7, num_samples=30)
# # 取得数据
# data = data.view(-1, in_shape).to(device)
# data, targets = data.to(device), targets.to(device)
# print(data.shape)
data, targets, k = get_negative_samples(target_class=8, num_samples=30)
for epochs in range(1):

    # 取得数据
    data = data.view(-1, in_shape).to(device)
    data, targets = data.to(device), targets.to(device)
    # print(k)
    optimizer1.zero_grad()
    outputs = net(data)
    eps = 1e-6
    dist = torch.sum((outputs - C) ** 2, dim=1)
    print(dist.shape)
    print(targets.shape)
    losses = (dist + eps) ** targets.float()
    loss = torch.mean(losses)
    print(f"net loss for epoch{epochs} : {loss}")
    loss.backward()
    optimizer1.step()


model = update_ae_net_from_net(net, model)


# 利用函数DASVDD_test对模型进行测试，其中传入的参数分别为：模型model、编码中心C、输入形状in_shape、Gamma值、测试数据集test_loader、标签集labels、损失函数criterion、C
# score, loss = DASVDD_test(model, C, in_shape, Gamma, test_loader, labels, criterion, C)
score, loss, normal_scores, anomaly_scores = DASVDD_test(model, C, in_shape, Gamma, test_loader, labels, criterion, C)
print(f"test auc:{score}")


main_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
all_bins = []
for i in range(len(main_bins)-1):
    all_bins.extend(np.linspace(main_bins[i], main_bins[i+1], 11)[:-1])
all_bins.append(1.0)

# Discretize the scores
normal_scores_binned = np.digitize(normal_scores, all_bins, right=True)
anomaly_scores_binned = np.digitize(anomaly_scores, all_bins, right=True)

# Convert bin indices back to actual bin values
normal_scores_binned = [all_bins[min(i, len(all_bins)-1)] for i in normal_scores_binned]
anomaly_scores_binned = [all_bins[min(i, len(all_bins)-1)] for i in anomaly_scores_binned]

# # Plot the binned histograms and save the figure
# plt.figure(figsize=(10, 6))
# plt.hist(normal_scores_binned, alpha=0.5, color='g', label='Normal Scores', bins=all_bins, align='mid', rwidth=0.9)
# plt.hist(anomaly_scores_binned, alpha=0.5, color='r', label='Anomaly Scores', bins=all_bins, align='mid', rwidth=0.9)
# plt.title('Distribution of Normal and Anomaly Scores')
# plt.xlabel('Score')
# plt.ylabel('Frequency')
# plt.xticks(main_bins)  # Set x-axis ticks
# plt.yticks([])         # Hide y-axis ticks
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Save the figure to a file
# file_path = "score_distribution_detailed.png"
# plt.savefig(file_path)
# plt.close()


# Calculate histogram values for normal and anomaly scores
normal_hist_values, normal_hist_edges = np.histogram(normal_scores_binned, bins=all_bins)
anomaly_hist_values, anomaly_hist_edges = np.histogram(anomaly_scores_binned, bins=all_bins)

# Calculate the center of each bin
normal_hist_centers = (normal_hist_edges[:-1] + normal_hist_edges[1:]) / 2
anomaly_hist_centers = (anomaly_hist_edges[:-1] + anomaly_hist_edges[1:]) / 2

# Plot the histograms and the curves connecting the top points
plt.figure(figsize=(12, 7))

# Plot histograms
plt.hist(normal_scores_binned, alpha=0.5, color='g', label='Normal Scores', bins=all_bins, align='mid', rwidth=0.9)
plt.hist(anomaly_scores_binned, alpha=0.5, color='#FFC0CB', label='Anomaly Scores', bins=all_bins, align='mid', rwidth=0.9)

# Plot curves connecting the top points
plt.plot(normal_hist_centers, normal_hist_values, color='g', marker='o')
plt.plot(anomaly_hist_centers, anomaly_hist_values, color='#FFC0CB', marker='o')

# Further plot customizations
plt.title('Distribution of Normal and Anomaly Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.xticks(main_bins)  # Set x-axis ticks
plt.yticks([])         # Hide y-axis ticks
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the figure to a file
file_path = "score_distribution_with_curve.png"
plt.savefig(file_path)
plt.close()





# data, targets, _ = get_negative_samples(target_class=8, num_samples=30)
# data = data.view(-1, in_shape).to(device)
# data, targets = data.to(device), targets.to(device)

# optimizer1.zero_grad()

# outputs = net(data)
# eps = 1e-6
# dist = torch.sum((outputs - C) ** 2, dim=1)
# # losses = (dist + eps) ** targets.float()
# losses = torch.where(targets == 1, dist, ((dist + eps) ** targets.float()))
# loss = torch.mean(losses)
# # print(f"net loss for epoch{epoch1} : {loss}")
# loss.backward()
# optimizer1.step()
# #
