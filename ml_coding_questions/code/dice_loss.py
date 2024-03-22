# https://zhuanlan.zhihu.com/p/269592183

# Dice loss is from VNet for Image Segmentation
# It is useful for class imbalance

# dice = 2 * TP / (2 * TP + FP + FN), which is the same as F1 score
# dice_loss = 1 - dice
# So optimzie dice_loss = optimzie F1 score

import torch

def dice_loss(target, predictive, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
