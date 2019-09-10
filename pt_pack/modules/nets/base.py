# coding=utf-8
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, kaiming_normal_
import logging
from pt_pack.meta import PtBase
from pt_pack.utils import try_get_attr, add_argument, func_params, str_bool, load_func_kwargs, load_func_params
from pt_pack.modules.layers import Layer


sys_logger = logging.getLogger(__name__)


__all__ = ['Net', 'LayerNet']


class Net(nn.Module, PtBase):
    type = 'pt_net'

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


class LayerNet(Net):
    prefix = 'layer_net'

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @classmethod
    def add_args(cls, group, params=None, sub_cls=None):
        add_argument(group, f'{cls.prefix_name()}_layer_names', type=str, nargs='*')
        group = cls.default_add_args(group, params)
        layer_cls_names = try_get_attr(params, f'{cls.prefix_name()}_layer_names', check=False)
        if layer_cls_names is not None:
            params = cls.collect_layer_params(layer_cls_names)
            for name, param in params.items():
                arg_type = param.annotation
                arg_type = str_bool if arg_type is bool else arg_type
                add_argument(group, f'{cls.prefix_name()}_layer_{name}s', type=arg_type, nargs='*')
        return group

    @classmethod
    def collect_layer_params(cls, layer_cls_names):
        params = dict()
        for layer_cls_name in layer_cls_names:
            layer_cls = Layer.load_cls(layer_cls_name)
            params.update(func_params(layer_cls.__init__))
        params = {name: param for name, param in params.items() if param.annotation in (int, str, bool, float)}
        return params

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        layer_cls_names = try_get_attr(params, f'{cls.prefix_name()}_layer_names')
        layer_clses = [Layer.load_cls(layer_cls_name) for layer_cls_name in layer_cls_names]
        layer_params = cls.collect_layer_params(layer_cls_names)
        layer_args = {name: try_get_attr(params, f'{cls.prefix_name()}_layer_{name}s') for name in layer_params.keys()}
        layer_args = {key: value for key, value in layer_args.items() if value is not None}
        layers = list()
        for idx, layer_cls in enumerate(layer_clses):
            layer_kwargs = {name: layer_arg[idx] for name, layer_arg in layer_args.items()}
            layer_kwargs = load_func_kwargs(layer_kwargs, layer_cls.__init__)
            layers.append(layer_cls(**layer_kwargs))
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        kwargs[f'{cls.prefix_name()}_layers'] = layers
        return cls.default_build(kwargs, controller=controller)











