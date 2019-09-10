# coding=utf-8
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import LambdaLR
from pt_pack.optimizers import Optimizer
from pt_pack.meta import PtBase
from typing import List
from pt_pack.utils import to_tensor, to_cuda
import random
import numpy as np
import itertools
import threading
from torch.cuda._utils import _get_device_index


__all__ = ['Cuda']


def _find_tensors(obj):  # pragma: no cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


def get_a_var(obj):  # pragma: no cover
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):  # pragma: no cover
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)

                # ---------------
                # CHANGE
                if module.training:
                    output = module.training_step(*input, **kwargs)
                else:
                    output = module.validation_step(*input, **kwargs)
                # ---------------

            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class LightningDataParallel(nn.DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        # we can assure that, so omit it
        # for t in chain(self.module.parameters(), self.module.buffers()):
        #     if t.device != self.src_device_obj:
        #         raise RuntimeError("module must have its parameters and buffers "
        #                            "on device {} (device_ids[0]) but found one of "
        #                            "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            # lightning
            if self.module.training:
                return self.module.step_train(*inputs[0], **kwargs[0])
            else:
                return self.module.step_eval(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class DataPrefetcher(object):
    def __init__(self, loader, cuda):
        self.loader = iter(loader)
        self.cuda = cuda
        self.stream = torch.cuda.Stream(device=cuda.master_device)
        self.current_sample = self.preload()

    def preload(self):
        with torch.cuda.stream(self.stream):
            try:
                sample = next(self.loader)
            except:
                return None
            sample = self.cuda.process_sample(sample, non_blocking=True)
        return sample

    def next(self):
        self.stream.synchronize()
        sample = self.current_sample
        self.current_sample = self.preload()
        return sample


class Cuda(PtBase):
    def __init__(self,
                 device_id: int = 0,
                 device_ids: List[int] = None,
                 seed: int = 1,
                 is_prefetch: bool = False,
                 dis_backend: str = None,
                 ):
        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        if device_ids is None:
            device_ids = [device_id]
        if device_ids is not None and device_id not in device_ids:
            device_id = device_ids[0]
        torch.cuda.set_device(device_id)
        self.master_device = torch.device('cuda', device_id)
        self.seed = seed
        self.device_ids = device_ids
        self.is_prefetch = is_prefetch
        self.init_seed(seed)
        self.loader = None
        if len(self.device_ids) > 0:
            self.use_dp = True if dis_backend is None else dis_backend == 'dp'
            self.use_ddp = False if dis_backend is None else dis_backend == 'ddp'
        super().__init__()

    def init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def process_model(self, model=None, criterion=None):
        if model is not None:
            if self.use_dp:
                model.cuda(self.master_device)
                model = LightningDataParallel(model, device_ids=self.device_ids)
            elif self.use_ddp:
                raise NotImplementedError()
        if criterion is not None:
            criterion.to(self.master_device)
        if model is not None and criterion is None:
            return model
        if model is None and criterion is not None:
            return criterion
        return model, criterion
        
    def process_sample(self, sample, non_blocking=False):
        sample = to_cuda(sample, self.master_device, non_blocking)
        sample = {key: to_tensor(value) for key, value in sample.items()}
        return sample

    def multi_gpu_scheduler(self, optimizers, warmup_epochs=3, lambda_func=None):
        scheduler = list()

        def labada_fn(epoch_idx):
            gpu_nums = len(self.device_ids)
            if warmup_epochs == 0:
                return gpu_nums
            if epoch_idx < warmup_epochs:
                return 1 + (gpu_nums - 1) * epoch_idx / (warmup_epochs-1)
            if epoch_idx >= warmup_epochs:
                return gpu_nums
        if isinstance(optimizers, Optimizer):
            optimizers = optimizers.optimizers
            optimizers = optimizers.values() if isinstance(optimizers, dict) else optimizers
            for optim in optimizers:
                scheduler.append(LambdaLR(optim, lambda_func or getattr(optim, 'policy_func', None) or labada_fn))
        else:
            scheduler.append(LambdaLR(optimizers, lambda_func or labada_fn))
        return scheduler

    def process_loader(self, loader):
        if self.is_prefetch:
            return DataPrefetcher(loader, self)
        return iter(loader)

    def init(self):
        controller = self.controller
        model, criterion = self.process_model(controller.model, controller.criterion)
        controller.module_dicts['model'] = model
        controller.module_dicts['criterion'] = criterion

    def before_epoch_train(self):
        self.loader = self.process_loader(self.controller.loaders['train'])

    def before_epoch_eval(self):
        self.loader = self.process_loader(self.controller.loaders['eval'])

    def step_train(self, message):
        if self.is_prefetch:
            batch_data = self.loader.next()
        else:
            batch_data = next(self.loader)
            batch_data = self.process_sample(batch_data)
        message['batch_data'] = {'name': 'batch_data', 'value': batch_data, 'tags': ('no_cpu', )}
        return message

    def step_eval(self, message):
        return self.step_train(message)









