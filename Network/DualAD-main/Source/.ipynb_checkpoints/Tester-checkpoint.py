import numpy as np
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import torchvision.transforms as transforms

# def DASVDD_test(model,C,in_shape,Gamma,test_loader,labels,criterion,ssim_loss,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

#   with torch.no_grad():
#     score = []
#     i = 0
#     for x_test in test_loader:
#       if isinstance(x_test, list):
#           x_test = x_test[0]
#       x_test =x_test.view(-1, in_shape).to(device)
#       x_test_hat,code_test = model(x_test)
#       loss = criterion(x_test_hat,x_test) + Gamma*torch.sum((code_test.to(device) - C) ** 2, dim=1)[0]
#       score.append(loss.to("cpu").item())
#       i+=1
#     return metrics.roc_auc_score(labels,score)*100,loss

def create_matrix_based_on_list(value, reference_list):
    # 获取列表的形状
    shape = np.array(reference_list).shape
    return np.full(shape, value)



def DASVDD_test(model, C, in_shape, Gamma, test_loader, labels, criterion, ssim_loss,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    with torch.no_grad():
        scores = []
        normal_scores = []
        anomaly_scores = []
        max_score = 0
        min_score = 0

        for i, x_test in enumerate(test_loader):
            if isinstance(x_test, list):
                x_test = x_test[0]
            x_test = x_test.view(-1, in_shape).to(device)
            x_test_hat, code_test = model(x_test)
            loss = criterion(x_test_hat, x_test) + 50 * torch.sum((code_test.to(device) - C) ** 2, dim=1)[0]
            if loss > max_score:
                max_score = loss
            if loss < min_score:
                min_score = loss
            scores.append(loss.to("cpu").item())
            
            # Classify scores based on the given labels
            if labels[i] == 0:  # Assuming 0 is the label for normal samples
                normal_scores.append(loss.to("cpu").item())
            else:
                anomaly_scores.append(loss.to("cpu").item())
        normal_scores = (normal_scores.cpu() - create_matrix_based_on_list(min_score, normal_scores))/(max_score - min_score)
        anomaly_scores = (anomaly_scores.cpu() - create_matrix_based_on_list(min_score, anomaly_scores))/(max_score - min_score)

        return metrics.roc_auc_score(labels, scores) * 100, loss, normal_scores, anomaly_scores



