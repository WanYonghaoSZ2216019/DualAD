import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms


def DASVDD_trainer(model, DASVDD_test, gan, discriminator, in_shape, code_size, C, train_loader, test_loader, labels, optimizer,optimizer_discriminator, update_center, criterion, criterion1, Gamma,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   num_epochs=300, K=0.9, verbosity=0):
    num_epochs = 5
    K = 0.9
    L1 = np.zeros(num_epochs)
    L2 = np.zeros(num_epochs)
    L3 = np.zeros(num_epochs)
    c_vals = np.zeros(num_epochs)
    max_score = 0
    max_epoch = 0
    count_epoch = 0
    for epoch in range(num_epochs):
        count_epoch = count_epoch + 1
        loss = 0
        aeloss = 0
        svddloss = 0
        for batch_features in train_loader:
            if isinstance(batch_features, list):
                batch_features = batch_features[0]
            batch_features = batch_features.view(-1, in_shape).to(device)
            Num_batch = int(np.ceil(K * batch_features.size()[0]))
            # 训练判别器
            optimizer_discriminator.zero_grad()
            real_labels = torch.ones(Num_batch, 1).to(device)
            fake_data, code = model(batch_features[:Num_batch, :])
            code = code.to(device)
            fake_labels = torch.zeros(Num_batch, 1).to(device)
            # print(batch_features.is_cuda)
            real_outputs, _ = discriminator(batch_features[:Num_batch, :])
            # print(real_outputs.is_cuda)
            fake_outputs, _ = discriminator(fake_data.detach())
            # print(real_outputs.shape)
            # print(real_labels.shape)
            d_loss_real = criterion1(real_outputs, real_labels).to(device)
            d_loss_fake = criterion1(fake_outputs, fake_labels).to(device)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_discriminator.step()
            optimizer_discriminator.zero_grad()

            # 训练生成器（通过GAN）
            optimizer.zero_grad()
            recon_data, disc_output, logits_x = gan(batch_features[:Num_batch, :])
            _, real_logits = discriminator(batch_features[:Num_batch, :])
            g_loss = criterion1(disc_output, real_labels)
            f_loss = criterion(logits_x, real_logits)

            # 添加重构损失 (如 MSE 计算像素级损失)
            R = torch.sum((code.to(device) - C) ** 2, dim=1)[0]
            train_loss = 0.01 * criterion(recon_data, batch_features[:Num_batch, :]) + Gamma * R + 0.96 * g_loss + 0.03 * f_loss
            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            aeloss += criterion(recon_data, batch_features[:Num_batch, :]).item()
            svddloss += R.item()
            _, c_code = model(batch_features[Num_batch:, :])
            center = torch.mean(c_code, axis=0)
            center_loss = criterion(C, center)
            center_loss.backward()
            update_center.step()
            update_center.zero_grad()
            c_vals[epoch] += C[0]
        score, loss,_,_ = DASVDD_test(model, C, in_shape, Gamma, test_loader, labels, criterion, C)
        if score > max_score:
            max_score = score
            max_epoch = count_epoch
        # print(f"test auc:{score}")
        c_vals[epoch] = c_vals[epoch] / len(train_loader)
        loss = loss / len(train_loader)
        aeloss = aeloss / len(train_loader)
        svddloss = svddloss / len(train_loader)
        L1[epoch] = loss
        L2[epoch] = aeloss
        L3[epoch] = svddloss
        if verbosity == 0:
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
        if verbosity == 2:
            return L1, L2, L3, c_vals
    print(f'score:{max_score}, epoch:{max_epoch}')

# In[ ]:




