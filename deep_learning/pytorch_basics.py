import numpy as np
import torch

data = [[1, 2], [3, 4], [5, 6]]
np_data =np.array(data)
print(np_data, type(np_data))

np_data_tensor = torch.from_numpy(np_data)
print(np_data_tensor, type(np_data_tensor))

print(torch.zeros(5))
print(torch.ones(2,2))
print(torch.rand(3, 3, 3))

zero_tensor = torch.zeros(3, 3)
ones_lise_zeros = torch.ones_like(zero_tensor)
print(zero_tensor.shape, ones_lise_zeros.shape)

print(zero_tensor[0])
print(ones_lise_zeros)

# ones_lise_zerosの全ての行の1列目
print(ones_lise_zeros[:, 0])

x = ones_lise_zeros.detach()
print(x)

x[:, 0] = 5
print(x)

rand1 = torch.rand(5)
rand2 = torch.rand(5)
print(torch.matmul(rand1, rand2))

rand1 = torch.rand(5, 5)
rand2 = torch.rand(5, 5)
print(torch.matmul(rand1, rand2))