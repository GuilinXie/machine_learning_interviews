import numpy as np
import torch
import torch.nn as nn

momentum = 0.9

# Solution 1: Compute batch_norm mannually
def batch_norm(feature, statistic_mean, statistic_var):
    feature_shape = feature.shape
    for i in range(feature_shape[1]):
        channel = feature[:, i, :, :]
        
        # 'mean' is the average of the numbers
        mean = channel.mean()           # mean 
        
        # Standard Deviation measures the spread of a data distribution
        var_1 = channel.var()           # population standard deviation 总体标准差
        std_2 = channel.std(ddof=1)     # sample standard deviation 样本标准差
        
        # Normalize the data in the same channel
        feature[:, i, :, :] = (feature[:, i, :, :] - mean) / np.sqrt(var_1 ** 2 + 1e-5)
        
        # Update statistic mean and standard deviation
        statistic_mean[i] = momentum * statistic_mean[i] + (1 - momentum) * mean
        statistic_var[i] = momentum * statistic_var[i] + (1 - momentum) * (std_2 ** 2)
        
        
    print(feature)
    print("statistic_mean : ", statistic_mean)
    print("statistic_var : ", statistic_var)
    

feature_array = np.random.randn(2, 2, 2, 2)
feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)

# Initialize mean and standard deviation
statistic_mean = [0.0, 0.0]
statistic_var = [1.0, 1.0]

# Manually compute the batch normalization
batch_norm(feature_array, statistic_mean, statistic_var)

# ------------------------------------------------------------------------------------------------

# Solution 2: Using torch.nn.BatchNorm2d
bn = nn.BatchNorm2d(num_features=2, eps=1e-5)
output = bn(feature_tensor)

print(output)
print('bn.running_mean : ', bn.running_mean)
print('bn.running_var : ', bn.running_var)


# ------------------------------------------------------------------------------------------------
# Solution 3: Implement a BatchNorm class similar to PyTorch's built-in Library
# https://zhuanlan.zhihu.com/p/269465213

def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):
    if not is_training:   # For prediction
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:                 # For training
        if len(x.shape) == 2:    # For fully-connected layer
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:                     # For conv layer
            mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # Normalize using the mean and var from this batch
        x_hat = (x - mean) / torch.sqrt(var + eps)
        
        # Update
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    
    x = gamma * x_hat + beta   # Rescaling and Offsetting
    return x, moving_mean, moving_var
             
            

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        
        # Check if it's fully connected or conv layer
        if num_dims == 2:       
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        
        # Initiate trainable parameters
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        # Initiate non-trainable parameters
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))
        
    def forward(self, x):
        # move data to the device same as x
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
            
        x, self.moving_mean, self.moving_var = batch_norm(self.training, 
                                                          x, 
                                                          self.gamma, 
                                                          self.beta, 
                                                          self.moving_mean,
                                                          eps=1e-5,
                                                          momentum=0.9)
        return x
        
        
            