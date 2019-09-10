# coding=utf-8
from .base import Optimizer
import torch.optim as optim
from typing import List
import adabound
import math


__all__ = ['Adam', 'Adamax', 'AdaBound']


class Adam(Optimizer):
    optim_cls = optim.Adam

    def __init__(self,
                 lr: float,
                 betas: List[float],
                 eps: float,
                 weight_decay: float,
                 max_epoch: int = math.inf,
                 epoch2val: int = 1,
                 clip_norm: float = -1.,
                 ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(max_epoch, epoch2val, clip_norm)


class Adamax(Optimizer):
    optim_cls = optim.Adamax

    def __init__(self,
                 lr: float,
                 betas: List[float],
                 eps: float,
                 weight_decay: float,
                 model,
                 max_epoch: int = math.inf,
                 epoch2val: int = 1,
                 clip_norm: float = -1.,
                 ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(model, max_epoch, epoch2val, clip_norm)


class AdaBound(Optimizer):
    optim_cls = adabound.AdaBound

    def __init__(self,
                 lr: float,
                 final_lr: float,
                 betas: List[float],
                 eps: float,
                 weight_decay: float,
                 model,
                 max_epoch: int = math.inf,
                 epoch2val: int = 1,
                 clip_norm: float = -1.,
                 ):
        self.lr = lr
        self.final_lr = final_lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(model, max_epoch, epoch2val, clip_norm)









