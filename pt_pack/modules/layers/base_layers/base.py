# coding=utf-8
import logging
from torch.nn.init import kaiming_normal_, kaiming_uniform_
import torch.nn as nn
from pt_pack.meta import PtBase

sys_logger = logging.getLogger(__name__)

__all__ = ['Layer']


# raise NotImplementedError()

class Layer(nn.Module, PtBase):

    def __init__(self):
        super().__init__()

    def reset_parameters(self, init='uniform'):
        sys_logger.info(f'Net {self.__class__} is resetting its parameters')
        if init.lower() == 'normal':
            init_params = kaiming_normal_
        elif init.lower() == 'uniform':
            init_params = kaiming_uniform_
        else:
            return
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init_params(m.weight)




