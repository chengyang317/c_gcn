from pt_pack.meta import PtBase
from pt_pack.utils import batch_messages_fusion
from collections import defaultdict
import math
import tqdm
import torch
import copy
import numpy as np


__all__ = ['Trainer']


class Trainer(PtBase):
    hooks = defaultdict(list)
    hook_locations = ('CALL', 'INIT', 'TRAIN', 'BEFORE_EPOCH_TRAIN', 'EPOCH_TRAIN', 'AFTER_EPOCH_TRAIN', 'EVAL',
                      'BEFORE_STEP_TRAIN', 'STEP_TRAIN', 'AFTER_STEP_TRAIN', 'BEFORE_EPOCH_EVAL', 'EPOCH_EVAL',
                      'AFTER_EPOCH_EVAL', 'BEFORE_STEP_EVAL', 'STEP_EVAL', 'AFTER_STEP_EVAL')

    def __init__(self,
                 max_epoch: int = -1,
                 epoch2val: int = 1,
                 step2prog: int = 100,
                 ):
        super().__init__()
        self.max_epoch = max_epoch if max_epoch != -1 else math.inf
        self.epoch2val = epoch2val
        self.step2prog = step2prog
        self._epoch_idx = None
        self.prog_bar = None
        self.mode = None
        self.global_step = 0
        self.loader = None
        self.batch_idx = None
        self.step_message = None
        self.train_epoch_message = None
        self.val_epoch_message = None
        self.running_messages = defaultdict(dict)
        self._module_types = None

    @property
    def module_types(self):
        if self._module_types is None:
            self._module_types = copy.deepcopy(self.controller.module_types)
            if 'trainer' in self._module_types:
                self._module_types.remove('trainer')
                self._module_types.insert(0, 'trainer')
        return self._module_types

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        return cls.default_build(params, controller)

    @classmethod
    def register_hook(cls, where, how='append'):
        assert where in cls.hook_locations

        def register(fn):
            if how == 'replace':
                cls.hooks[where] = [fn]
            else:
                cls.hooks[where].append(fn)
        return register

    def call(self, where, *args, **kwargs):
        results = list()
        assert where in self.hook_locations
        hook_fns = self.hooks[where]
        if len(hook_fns) == 0:
            results.append(self.module_iter_call(where.lower(), *args, **kwargs))
        else:
            for hook_fn in self.hooks[where]:
                results.append(hook_fn(self, *args, **kwargs))
        return results[-1]

    def __call__(self):
        return self.call('CALL')

    def __getattr__(self, item):
        return getattr(self.controller, item)

    @property
    def epoch_idx(self):
        if self._epoch_idx is None:
            self._epoch_idx = self.checkpoint.last_epoch
        return self._epoch_idx

    def module_call(self, module_type, func_name, *args, **kwargs):
        module = getattr(self, module_type)
        module_func = getattr(module, func_name)
        return module_func(*args, **kwargs)

    def module_iter_call(self, func_name, *args, **kwargs):
        for module_type in self.module_types:
            module = getattr(self, module_type)
            if module_type == 'model':
                module = self.real_model
            module_func = getattr(module, func_name.lower())
            module_func(*args, **kwargs)

    def message2cpu(self, message):
        for name, attr in message.items():
            if 'no_cpu' in attr.get('tags', ()):
                continue
            attr_value = attr['value']
            if torch.is_tensor(attr_value) and attr_value.is_cuda:
                attr['value'] = attr_value.item() if attr_value.dim() == 0 else attr_value.cpu()
        return message

    def set_tqdm_disc(self, message):
        if not self.batch_idx % self.step2prog == 0:
            return
        epoch_message = self.train_epoch_message if self.mode == 'train' else self.val_epoch_message
        disc = {}
        for name, attr in message.items():
            tags = attr.get('tags', ())
            if 'prog' not in tags:
                continue
            elif 'mean' not in tags:
                disc[f'{self.mode}_{name}'] = attr['value']
            else:
                disc[f'{self.mode}_{name}'] = np.array([message[name]['value'] for message in epoch_message[-100:]]).mean()
        self.prog_bar.set_postfix(**disc)

    def before_epoch_train(self):
        self.loader = self.loaders['train']
        self.prog_bar = tqdm.tqdm(range(len(self.loader)), position=0)
        self.mode = 'train'
        self.train_epoch_message = list()
        self._epoch_idx += 1

    def before_step_train(self, batch_idx):
        self.batch_idx = batch_idx
        self.real_model.global_step = self.global_step
        self.prog_bar.update(1)
        self.step_message = {
            'epoch_idx': {'name': 'epoch_idx', 'value': self.epoch_idx, 'tags': ['keep', 'prog']},
            'batch_idx': {'name': 'batch_idx', 'value': self.batch_idx, 'tags': ['keep']},
            'global_step': {'name': 'global_step', 'value': self.global_step, 'tags': ['keep']},
        }

    def step_train(self, message):
        loss_attr = message['loss'] or message['train_loss']
        loss_attr['value'].backward()
        message = self.message2cpu(message)
        self.step_message = message
        return message

    def after_step_train(self):
        step_message = {name: attr for name, attr in self.step_message.items() if 'keep' in attr.get('tags', ())}
        self.train_epoch_message.append(step_message)
        self.set_tqdm_disc(self.step_message)
        self.global_step += 1
        self.step_message = None

    @property
    def need_eval(self):
        return (self.epoch_idx + 1) % self.epoch2val == 0

    def before_epoch_eval(self):
        self.loader = self.loaders['eval']
        self.prog_bar = tqdm.tqdm(range(len(self.loader)), position=0)
        self.mode = 'eval'
        self.val_epoch_message = list()

    def before_step_eval(self, batch_idx):
        self.batch_idx = batch_idx
        self.prog_bar.update(1)
        self.step_message = {
            'batch_idx': {'name': 'batch_idx', 'value': self.batch_idx},
        }

    def step_eval(self, message):
        message = self.message2cpu(message)
        self.step_message = message
        return message

    def after_step_eval(self):
        step_message = {name: attr for name, attr in self.step_message.items() if 'keep' in attr.get('tags', ())}
        self.val_epoch_message.append(step_message)
        self.set_tqdm_disc(self.step_message)
        self.step_message = None

    def after_epoch_eval(self):
        fusion_message = batch_messages_fusion(self.val_epoch_message)
        self.running_messages['eval'][self.epoch_idx] = fusion_message
        self.val_epoch_message = None

    def after_epoch_train(self):
        fusion_message = batch_messages_fusion(self.train_epoch_message)
        self.running_messages['train'][self.epoch_idx] = fusion_message
        self.train_epoch_message = None


@Trainer.register_hook('CALL', 'replace')
def trainer_call(trainer: Trainer):
    trainer.call('INIT')
    trainer.call('TRAIN')


@Trainer.register_hook('TRAIN', 'replace')
def trainer_train(trainer: Trainer):
    while trainer.epoch_idx < trainer.max_epoch:
        trainer.call('BEFORE_EPOCH_TRAIN')
        trainer.call('EPOCH_TRAIN')
        if trainer.need_eval:
            trainer.call('EVAL')
        trainer.call('AFTER_EPOCH_TRAIN')


@Trainer.register_hook('EPOCH_TRAIN')
def trainer_epoch_train(trainer: Trainer):
    for batch_idx in range(len(trainer.loader)):
        # if batch_idx > 16:
        #     break
        trainer.call('BEFORE_STEP_TRAIN', batch_idx)
        trainer.call('STEP_TRAIN')
        trainer.call('AFTER_STEP_TRAIN')


@Trainer.register_hook('STEP_TRAIN')
def trainer_step_train(trainer: Trainer):
    step_message = trainer.cuda.step_train(trainer.step_message)
    step_message = trainer.model(step_message)
    step_message = trainer.criterion.step_train(step_message)

    # need reduce outputs for gpus > 1 , it will be fixed in the future.
    # can learn from pytorch-lightning

    step_message = trainer.step_train(step_message)
    step_message = trainer.optimizer.step_train(step_message)
    step_message = trainer.experiment.step_train(step_message)
    return step_message


@Trainer.register_hook('EVAL', 'replace')
def trainer_eval(trainer: Trainer):
    trainer.call('BEFORE_EPOCH_EVAL')
    trainer.call('EPOCH_EVAL')
    trainer.call('AFTER_EPOCH_EVAL')


@Trainer.register_hook('EPOCH_EVAL')
def trainer_epoch_eval(trainer: Trainer):
    for batch_idx in range(len(trainer.loader)):
        # if batch_idx > 16:
        #     break
        trainer.call('BEFORE_STEP_EVAL', batch_idx)
        trainer.call('STEP_EVAL')
        trainer.call('AFTER_STEP_EVAL')


@Trainer.register_hook('STEP_EVAL')
def trainer_step_eval(trainer: Trainer):
    step_message = trainer.cuda.step_eval(trainer.step_message)
    step_message = trainer.model(step_message)
    step_message = trainer.criterion.step_eval(step_message)

    # need reduce outputs for gpus > 1 , it will be fixed in the future.
    # can learn from pytorch-lightning

    step_message = trainer.step_eval(step_message)
    step_message = trainer.experiment.step_eval(step_message)
    return step_message










