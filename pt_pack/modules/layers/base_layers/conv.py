# coding=utf-8
from .base import Layer
from .common import Null, Act
from .norms import Norm2D, Norm1D
from .coord import Coord
from .se import Se
from .cond import Cond
import torch.nn as nn
import torch
import math

__all__ = ['Conv2D', 'Linear', 'KerneLinear']


class Conv2D(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 orders=('conv', 'norm', 'act'),
                 norm_type='batch',
                 norm_affine=True,
                 act_type='relu',
                 coord_type='default',
                 se_type='mix',
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.orders = orders
        for idx, order in enumerate(orders):
            if order == 'conv':
                if idx + 1 < len(orders) and orders[idx+1] == 'norm':
                    bias = False
                self.layers.append(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, dilation, groups, bias))
                in_dim = out_dim
            elif order == 'norm':
                self.layers.append(Norm2D(in_dim, norm_type, norm_affine))
            elif order == 'act':
                self.layers.append(Act(act_type))
            elif order == 'se':
                self.layers.append(Se(in_dim, type=se_type))
            elif order == 'coord':
                layer = Coord(type=coord_type)
                self.layers.append(layer)
                in_dim += layer.out_dim
            elif order == 'cond':
                self.layers.append(Cond())
            elif order == 'null':
                self.layers.append(Null())
            else:
                raise NotImplementedError(f'cant recognize order {order}')

    def forward(self, x, gamma=None, beta=None):
        for layer in self.layers:
            if isinstance(layer, Cond):
                x = layer(x, gamma, beta)
            else:
                x = layer(x)
        return x


class Linear(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 orders=('linear', 'norm', 'act'),
                 norm_type='batch',
                 norm_affine=True,
                 act_type='relu',
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.orders = orders
        for idx, order in enumerate(orders):
            if order == 'linear':
                if idx + 1 < len(orders) and orders[idx+1] == 'norm':
                    bias = False
                self.layers.append(nn.Linear(in_dim, out_dim, bias))
                in_dim = out_dim
            elif order == 'norm':
                self.layers.append(Norm1D(in_dim, norm_type, norm_affine))
            elif order == 'act':
                self.layers.append(Act(act_type))
            elif order == 'cond':
                self.layers.append(Cond())
            else:
                raise NotImplementedError(f'cant recognize order {order}')

    def forward(self, x, gamma=None, beta=None):
        for layer in self.layers:
            if isinstance(layer, Cond):
                x = layer(x, gamma, beta)
            else:
                x = layer(x)
        return x


class KerneLinear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 bias=True,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.Tensor(kernel_size, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(kernel_size, out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """

        :param x: b, obj_num, c
        :return:
        """
        b_size, obj_num, o_c = x.shape
        x = x.view(b_size*obj_num, -1)[:, None, None, :]
        out = x @ self.weight[None]  # m,k_size,1,out_dim
        out = out.squeeze()
        if self.bias is not None:
            out = out + self.bias[None]
        return out













