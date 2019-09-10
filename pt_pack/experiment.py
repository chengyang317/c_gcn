from pt_pack.meta import PtBase
from pt_pack.utils import to_path, try_set_attr, add_func_args, load_func_kwargs, to_namespace
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Experiment(PtBase):
    def __init__(self,
                 work_dir: str,
                 step2log: int = 1,
                 writer_kwargs=None
                 ):
        super().__init__()
        self.work_dir = to_path(work_dir)
        self.step2log = step2log
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)
        self.writer = SummaryWriter(**writer_kwargs)


    @classmethod
    def add_args(cls, group, params=None):
        super().add_args(group, params)
        add_func_args(group, SummaryWriter.__init__, cls.prefix_name())
        return group

    @classmethod
    def init_args(cls, params):
        work_dir = osp.join(params.work_dir, f'{params.proj_name}/{params.exp_name}/{params.exp_version}/experiment')
        try_set_attr(params, f'{cls.prefix_name()}_work_dir', work_dir)
        try_set_attr(params, f'{cls.prefix_name()}_log_dir', work_dir)

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        params = to_namespace(params)
        writer_kwargs = load_func_kwargs(params, SummaryWriter.__init__, cls.prefix_name())
        init_kwargs = load_func_kwargs(params, cls.__init__, cls.prefix_name())
        init_kwargs.update({'writer_kwargs': writer_kwargs})
        module = cls(**init_kwargs)
        controller.register_module(module)
        return module

    def log(self, step_responses):
        pass

    def is_log(self, batch_idx):
        return batch_idx % self.step2log == 0

    def after_step_train(self):
        trainer = self.controller.trainer
        if not self.is_log(trainer.batch_idx):
            return

        epoch_message = trainer.train_epoch_message
        step_message = epoch_message[-1]
        tf_disc = {}
        for name, attr in step_message.items():
            tags = attr.get('tags', ())
            if 'tf' not in tags:
                continue
            elif 'mean' not in tags:
                tf_disc[name] = attr['value']
            else:
                tf_disc[name] = np.array([message[name]['value'] for message in epoch_message[-100:]]).mean()
        for k, v in tf_disc.items():
            self.writer.add_scalar(f'train_step_{k}', v, trainer.global_step)

    def after_epoch_eval(self):
        trainer = self.controller.trainer
        message = trainer.running_messages['eval'][trainer.epoch_idx]
        for name, attr in message.items():
            if 'tf' in attr.get('tags', ()):
                self.writer.add_scalar(f'eval_epoch_{name}', attr['value'], trainer.epoch_idx)

    def after_epoch_train(self):
        trainer = self.controller.trainer
        message = trainer.running_messages['train'][trainer.epoch_idx]
        for name, attr in message.items():
            if 'tf' in attr.get('tags', ()):
                self.writer.add_scalar(f'train_epoch_{name}', attr['value'], trainer.epoch_idx)










