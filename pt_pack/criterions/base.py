# coding=utf-8
import torch.nn as nn
from pt_pack.meta import PtBase

__all__ = ['Criterion']


class Criterion(nn.Module, PtBase):
    loss_cls = None

    def __init__(self,
                 ):
        super().__init__()
        self.loss_l = self.loss_cls()

    def step_train(self, message):
        return self.forward(message)

    def step_eval(self, message):
        return self.forward(message)

