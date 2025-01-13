# coding=utf-8

import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import numpy as np

def kl_loss(y_pred, y_true):    
    y_pred = F.log_softmax(y_pred, dim=-1)
    losses = F.kl_div(y_pred, y_true, reduction='none').sum(dim=-1)
    return losses

def mae_nmse_loss(y_pred, y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()  
    MAE = mean_absolute_error(y_true, y_pred)
    nMSE = np.mean(np.square(y_true - y_pred)) / (y_pred.std() ** 2)
    res = MAE + nMSE
    return res