from torch.utils.data import Sampler
import torch
import numpy as np


class UniformSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self._num_samples = len(data_source)
        self._batch_size = batch_size

    @property
    def num_samples(self):
        return self._num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = len(self.data_source)
        niter = int(np.ceil(n / self._batch_size))
        ret = []
        for ii in range(niter):
            ret.extend(torch.randperm(n)[: self._batch_size])
        return iter(ret)
