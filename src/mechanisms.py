import torch
from torch.distributions.log_normal import LogNormal

class GaussianNoise(object):
    """
    Add gaussian noise to a point cloud.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        shape = data.pos.shape
        minimum = data.pos.min()
        maximum = data.pos.max()
        assert shape[-1] == 3
        noise = torch.randn(size=(shape[0],))

        data.pos[:,-1] += noise
        data.pos = torch.clamp(data.pos, min=minimum, max=maximum)
        return data

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.mean, self.std)
