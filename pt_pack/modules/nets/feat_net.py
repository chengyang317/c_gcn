# coding=utf-8
import torch.nn as nn
import torchvision
import logging
from .base import Net


sys_logger = logging.getLogger(__name__)


__all__ = ['Resnet']


class Resnet(Net):
    def __init__(self, name, pre_trained=True, req_layer_ids=(5,)):
        super().__init__()
        assert hasattr(torchvision.models, name)
        resnet = getattr(torchvision.models, name)(pre_trained)
        layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ]
        self.required_layer_idxs = req_layer_ids  # start from number one
        last_layer_idx = max(req_layer_ids)
        self.layers = nn.ModuleList([layers[idx] for idx in range(last_layer_idx)])

    def forward(self, x):
        outs_per_layer = []
        for layer in self.layers:
            x = layer(x)
            outs_per_layer.append(x)
        outs = [outs_per_layer[layer_idx-1] for layer_idx in self.required_layer_idxs]
        if len(outs) == 1:
            return outs[0]
        return outs

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params)














































