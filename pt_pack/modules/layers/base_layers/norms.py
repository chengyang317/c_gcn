# coding=utf-8
import torch
import torch.nn as nn
from .base import Layer
from pt_pack.utils import try_get_attr

__all__ = ['Norm1D', 'Norm2D']


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, affine=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            return x * self.weight + self.bias
        return x


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False, affine=True):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.affine:
            if self.last_gamma:
                self.weight.data.fill_(0)
            else:
                self.weight.data.fill_(1)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        if self.affine:
            return x * self.weight + self.bias
        return x


class SwitchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm3d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, D, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, D, H, W)
        return x * self.weight + self.bias


class CombineNorm2d(nn.Module):
    def __init__(self, in_dim, affine=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_dim//2, affine=affine)
        self.in_norm = nn.InstanceNorm2d(in_dim//2, affine=affine)

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.shape
        x0, x1 = torch.split(x, C // 2, dim=1)
        x0 = self.batch_norm(x0.contiguous())
        x1 = self.in_norm(x1.contiguous())
        x = torch.cat((x0, x1), dim=1)
        return x

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))


class Norm2D(Layer):
    def __init__(self, in_dim, norm_type='batch', norm_affine=True, feat_h=14):
        super().__init__()
        if norm_type == 'batch':
            self.norm_l = nn.BatchNorm2d(in_dim, affine=norm_affine)
        elif norm_type == 'instance':
            self.norm_l = nn.InstanceNorm2d(in_dim, affine=norm_affine)
        elif norm_type == 'layer':
            in_dim = (in_dim, feat_h, feat_h)
            self.norm_l = nn.LayerNorm(in_dim, elementwise_affine=norm_affine)
        elif norm_type == 'combine':
            self.norm_l = CombineNorm2d(in_dim, affine=norm_affine)
        elif norm_type == 'switch':
            self.norm_l = SwitchNorm2d(in_dim, affine=norm_affine)
        else:
            raise ImportError(f'Cant recognize norm type {norm_type}')

    def forward(self, x):
        return self.norm_l(x)

    def build(cls, params, cls_name=None, sub_cls=None):
        return cls(params.in_dim, try_get_attr(params, 'norm_type', 'batch'), try_get_attr(params, 'norm_affine', True))


class Norm1D(Layer):
    def __init__(self, in_dim, norm_type='batch', norm_affine=True):
        super().__init__()
        if norm_type == 'batch':
            self.norm_l = nn.BatchNorm1d(in_dim, affine=norm_affine)
        elif norm_type == 'instance':
            # self.norm_l = nn.InstanceNorm1d(in_dim, affine=norm_affine)
            self.norm_l = None
        elif norm_type == 'layer':
            self.norm_l = nn.LayerNorm(in_dim, elementwise_affine=norm_affine)
        elif norm_type == 'combine':
            self.norm_l = None
        elif norm_type == 'switch':
            self.norm_l = SwitchNorm1d(in_dim, affine=norm_affine)
        else:
            raise ImportError(f'Cant recognize norm type {norm_type}')

    def forward(self, x):
        if self.norm_l is not None:
            return self.norm_l(x)
        return x

    def build(cls, params, cls_name=None, sub_cls=None):
        return cls(params.in_dim, try_get_attr(params, 'norm_type', 'batch'), try_get_attr(params, 'norm_affine', True))