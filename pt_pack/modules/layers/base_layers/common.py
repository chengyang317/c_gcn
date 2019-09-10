# coding=utf-8
from .base import Layer
from pt_pack.utils import try_get_attr
import torch.nn as nn

__all__ = ['Null', 'Flatten', 'Act']


class Null(Layer):
    def forward(self, x):
        return x


class Flatten(Layer):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Act(Layer):
    valid_types = ('relu', 'swish')

    def __init__(self, type='relu'):
        super(Act, self).__init__()
        assert type in self.valid_types
        self.type = type
        if type == 'relu':
            self.act_l = nn.ReLU(inplace=True)
        elif type == 'swish':
            self.act_l = lambda x: x * x.sigmoid()
        elif type is None:
            self.act_l = None

    def __repr__(self):
        return f"act {self.type}"

    def forward(self, x):
        if self.act_l is not None:
            return self.act_l(x)
        return x

    def build(cls, params, cls_name=None, sub_cls=None):
        return cls(try_get_attr(params, 'type', 'relu'))











