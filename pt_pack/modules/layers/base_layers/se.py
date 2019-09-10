# coding=utf-8
from torch import nn
from .base import Layer
from pt_pack.utils import try_get_attr
import torch


__all__ = ['Se']


class ChannelSe(nn.Module):
    def __init__(self, in_dim, widen=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_n = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // widen, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // widen, in_dim, kernel_size=1),
        )
        self.sigmoid_l = nn.Sigmoid()

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_avg = self.fc_n(x_avg)
        x_max = self.fc_n(x_max)
        x_sta = x_avg + x_max
        x = x * self.sigmoid_l(x_sta)
        return x


class SpatialSe(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv_n = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            # nn.BatchNorm2d(1),
        )
        self.sigmoid_l = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x_avg = x.mean(dim=1, keepdim=True)
        x_max, _ = x.max(dim=1, keepdim=True)
        x_sta = self.conv_n(torch.cat((x_avg, x_max), 1))
        x = x * self.sigmoid_l(x_sta)
        return x


class Se(Layer):
    def __init__(self, in_dim=None, widen=4, type='mix'):
        super().__init__()
        assert type in ('channel', 'spatial', 'mix')
        if type == 'mix':
            self.layer = nn.Sequential(
                ChannelSe(in_dim, widen),
                SpatialSe()
            )
        elif type == 'channel':
            self.layer = ChannelSe(in_dim, widen)
        else:
            self.layer = SpatialSe()

    def forward(self, x):
        return self.layer(x)

    def build(cls, params, cls_name=None, sub_cls=None):
        return cls(try_get_attr(params, 'in_dim', None), try_get_attr(params, 'widen', 2), try_get_attr(params, 'type', 'mix'))



