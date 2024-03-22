import torch
import numpy as np

# y_i = x_1 + x_2 + x_3 + ...+ x_i

# Solution 1: Using torch.cumsum
# a = torch.randn(10)
a = torch.tensor([1, 2, 3, 4])
b = torch.cumsum(a, dim=0)

# Solution2: Using number.cumsum
a = np.array([[1, 2, 3], [4, 5, 6]])

b = np.cumsum(a)    # array([1, 3, 6, 10, 15, 21])
c = np.cumsum(a, dtype=float)   # array([1., 3., 6., 10., 15., 21.])
d = np.cumsum(a, axis=0)    # array([[1, 2, 3], [5, 7, 9]])
e = np.cumsum(a, axis=1)    # array([[1, 3, 6], [4, 9, 15]])

# Solution 3: Python implementation
def cumulative_sum(data):
    cum_sum = []
    length = len(data)
    cum_sum = [sum(data[0: x+1]) for x in range(0, length)]
    return cum_sum



if __name__ == "__main__":
    print(cumulative_sum([1, 2, 3, 4, 5]))
    
    
    
    


