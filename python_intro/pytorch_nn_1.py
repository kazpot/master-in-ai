import torch

def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))

# set the random seed
torch.manual_seed(7)

# features are 5 random normal variables
features = torch.randn((1, 5))

# weights, random normal variables
weights = torch.randn_like(features)

# bias
bias = torch.randn((1, 1))

y = activation(torch.mm(features, weights.reshape(5, 1)) + bias)
print(y)