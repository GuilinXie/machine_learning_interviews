# focal_loss(p_t) = - alpha * [(1 - p_t) ** gamma] * log(p_t)
# usually, alpha = 0.25, gamma = 2
# When p_t -> 1, then (1 - p_t) ** gamma -> 0. This means that the weight of the loss decreases for these well-classified observations.
# when p_t -> 0, then (1 - p_t) ** gamma -> 1. This means that the weight of the loss doesn't change for these wrongly classified observations.

# CE loss (crossentropy loss)
# CE(p_t) = - log(p_t)

# Binary classification: sigmoid
# Multi-classification: softmax


import torch
import torch.nn.functional as F

# Focal Loss for Binary classification
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, predict, target):
        # Calculate Binary probability 
        pt = torch.sigmoid(predict)
        
        # BCELoss = -y * log(p_t) - (1 - y) * log(1 - p_t)
        # Apply Focal Loss to BCELoss
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
            -(1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# Focal Loss for Multi-classification
class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.calss_num = class_num
        
    def forward(self, predict, target):
        
        pt = F.softmax(predict, dim=1)      # Get probability
        class_mask = F.one_hot(target, self.class_num)  # Convert target to one-hot encoding
        ids = target.view(-1, 1)                # reshape target from row to column e.g. From [2, 3, 4] to [[2], [3], [4]]
        alpha = self.alpha[ids.data.view(-1)]   # alpha weights for each target observations, using ids as index of self.alpha
        probs = (pt * class_mask).sum(dim=1).view(-1, 1)    # using one-hot as mask, getting the pt in the right target position
        log_p = probs.log()
        
        # Apply Focal Loss to MultiBCELoss
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss        
        
            