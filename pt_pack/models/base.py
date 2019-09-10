# coding=utf-8
import torch.nn as nn
from typing import Dict, List
from pt_pack.meta import PtBase
from pt_pack.utils import add_argument, try_get_attr, load_func_kwargs, load_func_params, cls_name_trans
from pt_pack.modules import Net
import torch

__all__ = ['Model', 'NetModel']


class Model(nn.Module, PtBase):
    def __init__(self):
        super().__init__()
        self.global_step = None

    @classmethod
    def get_input(cls, sample):
        kwargs = load_func_kwargs(sample, cls.forward)
        return kwargs

    def set_mode(self, is_train=None):
        if is_train:
            self.train()
        else:
            self.eval()

    def build_optimizers(self):
        pass

    def save_model_structure(self, model):
        model = model.module if isinstance(model, nn.DataParallel) else model
        model_name = cls_name_trans(model.__class__.__name__)
        checkpoint = self.controller.checkpoint
        model_struct_file = checkpoint.work_dir.joinpath(f'{model_name}_model_structure')
        with model_struct_file.open('w') as fd:
            fd.write(str(model))

    def before_epoch_train(self):
        self.set_mode(is_train=True)

    def step_train(self, message):
        model_input = self.get_input(message['batch_data']['value'])
        model_output = self.forward(**model_input)
        message.update(model_output)
        return message

    def before_epoch_eval(self):
        self.set_mode(is_train=False)

    def step_eval(self, message):
        model_input = self.get_input(message['batch_data']['value'])
        with torch.no_grad():
            model_output = self.forward(**model_input)
        message.update(model_output)
        return message


class NetModel(Model):
    net_names = None

    def __init__(self, nets):
        super().__init__()
        assert isinstance(nets, (Dict, List))
        assert self.net_names is not None, 'Define net_names when define your model'
        self.net_dicts = nets if isinstance(nets, Dict) else {net.__class__.__name__: net for net in nets}
        assert len(self.net_names) == len(self.net_dicts) and all([name in self.net_dicts for name in self.net_names])

    @property
    def nets(self):
        return [self.net_dicts[name] for name in self.net_names]

    @classmethod
    def add_args(cls, group, params=None, sub_cls=None):
        for net_name in cls.net_names:
            add_argument(group, f'{cls.prefix_name()}_{net_name}_cls', type=str)
            net_cls = Net.load_cls(try_get_attr(params, f'{cls.prefix_name()}_{net_name}_cls', check=False))
            if net_cls is not None:
                net_cls.add_args(group, params)
        return group

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        for net_name in cls.net_names:
            net = Net.build(params, sub_cls=try_get_attr(params, f'{cls.prefix_name()}_{net_name}_cls'),
                            controller=controller)
            kwargs[f'{cls.prefix_name()}_{net_name}'] = net
        return cls.default_build(kwargs, controller=controller)































































