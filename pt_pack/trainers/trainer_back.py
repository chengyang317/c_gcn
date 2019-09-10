import torch
import logging
import torch.nn as nn
import math
from tqdm import tqdm
from pt_pack.checkpoint import Checkpoint
from pt_pack.models import Model
from pt_pack.loggers import Logger
from pt_pack.criterions import Criterion
from pt_pack.optimizers import Optimizer
from pt_pack.cuda import Cuda
from pt_pack.parsers.base import Parser
from pt_pack.datasets import Dataset
from typing import List, Dict
from pt_pack.meta import PtBase
from collections import defaultdict
from torch.utils.data import DataLoader
from collections import namedtuple


sys_logger = logging.getLogger(__name__)


__all__ = ['Trainer', 'Evaluator']


EpochResponse = namedtuple('EpochResponse', ['epoch_idx', 'train_responses', 'eval_responses'])
StepResponse = namedtuple('StepResponse', ['step_idx', 'log'])


class Controller(object):
    def __init__(self, args):
        self.args = args
        self.name = getattr(args, 'proj_name', 'default')
        self._modules = defaultdict(list)
        self._last_epoch = None

    @classmethod
    def build(cls, **kwargs):
        parser = Parser.build(**kwargs)
        return cls(parser.args)

    def register_module(self, module: PtBase):
        module.controller = self
        self._modules[module.prefix_name()].append(module)



class Evaluator(object):
    def __init__(self,
                 name,
                 args=None,
                 model=None,
                 loaders=None,
                 cuda=None,
                 checkpoint=None,
                 ):
        self.name = name
        self.args = args
        self.model: Model = model
        self.loaders = loaders
        self.cuda: Cuda = cuda
        self.checkpoint: Checkpoint = checkpoint

    @staticmethod
    def init_process(model, cuda, checkpoint):
        model = cuda.process_model(model)
        checkpoint.load_checkpoint(model)
        return model

    @classmethod
    def build(cls, **kwargs):
        parser = Parser.build(**kwargs)
        args = parser.args
        loaders = [Dataset.build_loader(args, split) for split in args.dataset_splits]
        model = Model.build(args)
        cuda = Cuda.build(args)
        checkpoint = Checkpoint.build(args)
        model = cls.init_process(model, cuda, checkpoint)
        return cls(args.proj_name, args, model, loaders, cuda, checkpoint)

    def __call__(self, evaluate_fn=None):
        if evaluate_fn is not None:
            return evaluate_fn(self)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_mode(False)
        else:
            self.model.set_mode(False)

        assert len(self.loaders) == 1
        raise NotImplementedError()


class Trainer(Controller):

    hooks = defaultdict(list)
    hook_locations = ('CALL', 'INIT', 'BEFORE_TRAIN', 'TRAIN', 'TRAIN_HOOKS', 'EVAL', 'EVAL_HOOKS', 'AFTER_TRAIN')

    def __init__(self,
                 args,
                 model: Model = None,
                 cuda: Cuda = None,
                 loggers: Dict[str, Logger] = None,
                 optimizer: Optimizer = None,
                 criterion: Criterion = None,
                 checkpoint: Checkpoint = None,
                 loaders: Dict[str, DataLoader] = None
                 ):
        super().__init__(args)
        self.model = model or Model.build(args, controller=self)
        self.cuda = cuda or Cuda.build(args, controller=self)
        self.loggers = loggers or Logger.build_loggers(args, controller=self)
        self.optimizer = optimizer or Optimizer.build(args, controller=self)
        self.criterion = criterion or Criterion.build(args, controller=self)
        self.checkpoint = checkpoint or Checkpoint.build(args, controller=self)
        self.loaders = loaders or Dataset.build_loaders(args, controller=self)
        self.response: List[EpochResponse] = []

    @classmethod
    def register_hook(cls, where, how='append'):
        assert where in cls.hook_locations

        def register(fn):
            if how == 'replace':
                cls.hooks[where] = [fn]
            else:
                cls.hooks[where].append(fn)
        return register

    def call(self, where):
        assert where in self.hook_locations
        for hook_fn in self.hooks[where]:
            hook_fn(self)

    def __call__(self):
        return self.call('CALL')

    @property
    def real_model(self) -> Model:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    @property
    def max_epoch(self):
        max_epoch = self.optimizer.max_epoch or math.inf
        return max_epoch

    @property
    def last_epoch(self):
        if self._last_epoch is None:
            self._last_epoch = self.checkpoint.last_epoch
        return self._last_epoch

    @property
    def is_verbose(self):
        if self.args is not None:
            return getattr(self.args, 'verbose', None)
        return None

    def need_val(self, epoch_idx):
        return epoch_idx % self.optimizer.epoch2val == 0


@Trainer.register_hook('CALL', 'replace')
def trainer_call(trainer: Trainer):
    trainer.call('INIT')
    epoch_idx = trainer.last_epoch + 1
    while epoch_idx < trainer.max_epoch:
        trainer.call('BEFORE_TRAIN')
        trainer.call('TRAIN')
        trainer.call('EVAL')
        trainer.call('AFTER_TRAIN')
        trainer._last_epoch = epoch_idx
        epoch_idx += 1


@Trainer.register_hook('INIT', 'replace')
def trainer_init(trainer: Trainer):
    trainer.model = trainer.cuda.process_model(trainer.model)
    trainer.checkpoint.load_checkpoint(trainer.model)
    trainer.cuda.process_model(criterion=trainer.criterion)
    trainer.optimizer.build_optimizers(trainer.model)
    trainer.response = list()


@Trainer.register_hook('BEFORE_TRAIN')
def before_train(trainer: Trainer):
    epoch_response = EpochResponse(trainer.last_epoch+1, [], [])
    trainer.response.append(epoch_response)


@Trainer.register_hook('TRAIN')
def train(trainer: Trainer):
    epoch_response = trainer.response[-1]
    trainer.real_model.set_mode(is_train=True)
    loader = trainer.loaders['train']
    pbar = tqdm(range(len(loader)), desc='Training...')
    cuda_loader = trainer.cuda.process_loader(loader)

    for step_idx in pbar:
        if trainer.cuda.is_prefetch:
            sample = cuda_loader.next()
        else:
            sample = next(cuda_loader)
            sample = trainer.cuda.process_sample(sample)
        model_input = trainer.real_model.get_input(sample)
        model_output = trainer.model(**model_input)

        loss, log = trainer.criterion(model_output, sample)
        trainer.optimizer.backward(loss, epoch_response.epoch_idx)
        epoch_response.train_responses.append(StepResponse(step_idx, log))
        trainer.call('TRAIN_HOOKS')
    pbar.close()


@Trainer.register_hook('TRAIN_HOOKS')
def after_step_train(trainer):
    epoch_response = trainer.response[-1]
    step_response = epoch_response.train_responses[-1]
    for split, logger in trainer.loggers.items():
        logger.forward(step_response.log[split], step_response.step_idx, epoch_response.epoch_idx, is_train=True,
                       need_log=trainer.is_verbose)


@Trainer.register_hook('EVAL')
def eval(trainer: Trainer):
    epoch_response = trainer.response[-1]
    if not trainer.need_val(epoch_response.epoch_idx):
        return
    trainer.real_model.set_mode(False)
    loader = trainer.loaders.get('eval', None)
    if loader is None:
        sys_logger.info(f'No eval loader finds')
        return
    pbar = tqdm(range(len(loader)), desc='Evaluating...')
    cuda_loader = trainer.cuda.process_loader(loader)
    for step_idx in pbar:
        sample = cuda_loader.next()
        with torch.no_grad():
            model_output = trainer.model(**trainer.real_model.get_input(sample))
            _, log = trainer.criterion(model_output, sample)
        epoch_response.eval_responses.append(StepResponse(step_idx, log))
        trainer.call('EVAL_HOOKS')
    pbar.close()


@Trainer.register_hook('EVAL_HOOKS')
def after_step_train(trainer):
    epoch_response = trainer.response[-1]
    step_response = epoch_response.eval_responses[-1]
    for split, logger in trainer.loggers.items():
        logger.forward(step_response.log[split], step_response.step_idx, epoch_response.epoch_idx, is_train=False,
                       need_log=False)


@Trainer.register_hook('AFTER_TRAIN')
def after_train(trainer: Trainer):
    epoch_response = trainer.response[-1]
    for split, logger in trainer.loggers.items():
        logger.epoch_update(epoch_response.epoch_idx, is_train=True)
        logger.epoch_update(epoch_response.epoch_idx, is_train=False)
    if 'acc' in trainer.loggers:
        val_acc = trainer.loggers['acc'].metric.epoch_avg(epoch_response.epoch_idx, is_train=False)
    else:
        val_acc = None
    trainer.checkpoint.save_checkpoint(epoch_response.epoch_idx, trainer.real_model, val_acc)
























