# Layer Norm is used in Transformer Block
# Input size = (batch_size, token_num, dim)
# Normalized in the last dim dimension: nn.LayerNorm(normalized_shape = dim)

# Example:
# Input: shape = [2, 4, 3]
# batch_size = 2, token_num = 4, dim = 3
# nn.LayerNorm(normalized_shape = 3)

import numpy as np
import torch.nn as nn
import torch

# Solution 1: Compute layer_norm mannually
def layer_norm(feature):
    b, token_num, dim = feature.shape
    feature = feature.reshape(-1, dim)
    for i in range(b * token_num):
        mean = feature[i, :].mean()
        var = feature[i, :].var()
        # print(mean, var)
        feature[i, :] = (feature[i, :] - mean) / np.sqrt(var + 1e-5)
    print(feature.reshape(b, token_num, dim))

 
feature_array = np.array([[[[1, 0], [0, 2]],
                          [[3, 4], [1, 2]],
                          [[2, 3], [4, 2]]],
                         [[[1, 2], [-1, 0]],
                          [[1, 2], [3, 5]],
                          [[1, 4], [1, 5]]]], dtype=np.float32)    # (2, 3, 2, 2)

# feature_array = np.random.randn(2, 3, 2, 2)
feature_array = feature_array.reshape(2, 3, -1).transpose(0, 2, 1)
feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)

layer_norm(feature_array)

# ------------------------------------------------------------------------------------------------

# Solution 2: Using nn.LayerNorm
ln = nn.LayerNorm(normalized_shape=3)
output = ln(feature_tensor)
print(output)

