# coding=utf-8
from pt_pack.meta import PtBase
from pt_pack.utils import cls_name_trans, try_get_attr, add_func_args, init_func_args, load_func_kwargs
import torch.optim as optim
import copy
import torch.nn as nn

__all__ = ['Optimizer']


class Optimizer(PtBase):

    def __init__(self,
                 type: str = 'adam',
                 batch2grad: int = 1,
                 grad_clip: int = -1,
                 clip_norm: float = None,
                 is_print_grads: bool = False,
                 optim_kwargs=None
                 ):
        super().__init__()
        self.optim_type = type
        self.batch2grad = batch2grad
        self.grad_clip = grad_clip
        self.clip_norm = clip_norm
        self.is_print_grads = is_print_grads
        self.optim_kwargs = optim_kwargs
        self.optimizers = None

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        opt_type = try_get_attr(params, f'{cls.prefix_name()}_type', None, check=False)
        opt_cls = getattr(optim, opt_type, None)
        assert opt_cls is not None
        optim_kwargs = load_func_kwargs(params, opt_cls.__init__, cls.prefix_name())
        init_kwargs = load_func_kwargs(params, cls.__init__, cls.prefix_name())
        init_kwargs.update({'optim_kwargs': optim_kwargs})
        return cls(**init_kwargs)

    @classmethod
    def add_args(cls, group, params=None):
        super().add_args(group, params)
        opt_type = try_get_attr(params, f'{cls.prefix_name()}_type', None, check=False)
        if opt_type is None:
            return group
        opt_cls = getattr(optim, opt_type, None)
        assert opt_cls is not None
        add_func_args(group, opt_cls.__init__, prefix=cls.prefix_name())
        return group

    @property
    def optim_cls(self):
        return getattr(optim, self.optim_type, None)

    @classmethod
    def init_args(cls, params):
        params = super().init_args(params)
        opt_type = try_get_attr(params, f'{cls.prefix_name()}_type', None, check=False)
        if opt_type is None:
            return params
        opt_cls = getattr(optim, opt_type, None)
        assert opt_cls is not None
        return init_func_args(opt_cls.__init__, params, cls.prefix_name())

    def default_build_optimizers(self, model):
        params = filter(lambda p: p.requires_grad, model.parameters())
        optim_kwargs = copy.deepcopy(self.optim_kwargs)
        optim_kwargs.update({'params': params})
        optimizer = self.optim_cls(**optim_kwargs)
        optimizers = [{'name': cls_name_trans(model.__class__.__name__), 'optimizer': optimizer, 'valid_fn': None}]
        return {'optimizers': optimizers, 'schedulers': None}

    def build_optimizers(self):
        model = self.real_model
        if 'build_optimizers' in model.__dict__:
            optimizers = model.build_optimizers(self)
        else:
            optimizers = self.default_build_optimizers(model)
        assert isinstance(optimizers, dict)
        self.optimizers = optimizers

    def install_policy(self, policy):
        assert isinstance(policy, dict) and isinstance(self.optimizers, dict)
        for name, fn in policy.items():
            optimizer = self.optimizers[name]
            setattr(optimizer, 'policy_func', fn)

    def init(self):
        self.build_optimizers()

    def before_epoch_train(self):
        if getattr(self.optimizers, 'schedulers', None) is not None:
            for scheduler in self.optimizers['schedulers']:
                scheduler.step()

    def is_grad(self, batch_idx):
        return True if (batch_idx + 1) % self.batch2grad == 0 else False

    def clip_grad(self):
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm(self.real_model.parameters(), self.grad_clip)

    @property
    def real_model(self):
        return self.controller.real_model

    def print_grads(self):
        if self.is_print_grads:
            for param in self.real_model.parameters():
                print(param.grad.float().sum())

    def step_train(self, message):
        self.print_grads()
        if not self.is_grad(message['batch_idx']['value']):
            return message
        self.clip_grad()
        epoch_idx = message['epoch_idx']['value']
        optimizers = self.optimizers['optimizers']
        valid_optimizers = filter(lambda x: True if getattr(x, 'valid_fn', None) is None else x['valid_fn'](epoch_idx), optimizers)
        for optimizer in valid_optimizers:
            optimizer['optimizer'].step()
        for optimizer in optimizers:
            optimizer['optimizer'].zero_grad()
        return message








