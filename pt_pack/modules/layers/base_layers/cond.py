# coding=utf-8
from .base import Layer


__all__ = ['Cond']


class Cond(Layer):
    def __init__(self):
        super().__init__()

    def expand_shape(self, gamma, x=None):
        if x is None and gamma.dim() == 2:
            return gamma.view(*gamma.shape[:2], 1, 1)
        elif x is not None:
            un_dim = 1 if x.dim() == 3 else -1
            for _ in range(gamma.dim(), x.dim()):
                gamma = gamma.unsqueeze(un_dim)
        return gamma

    def forward(self, x, gamma, beta=None):
        origin_shape = x.shape
        x = x * self.expand_shape(gamma, x) + self.expand_shape(beta, x)
        return x.view(*origin_shape)



