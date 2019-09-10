# coding=utf-8
from collections import defaultdict
import json
from .visdom import VisdomEnv, LineWin
from termcolor import colored
import numpy as np
import logging
import copy
from pt_pack.utils import try_set_attr, try_get_attr, to_path, load_func_kwargs, to_namespace
from pt_pack.meta import PtBase
from typing import List, Dict


sys_logger = logging.getLogger(__name__)


__all__ = ['Logger', 'LoggerGroup', 'PrintLogger', 'VisdomLogger']


class Metric(object):
    def __init__(self, name, logger_dir):
        self.name = name
        self.logger_dir = to_path(logger_dir)
        self.json_file = self.logger_dir.joinpath(f'{name}.json')
        self.histories = dict() if not self.json_file.exists() else json.load(self.json_file.open())
        self.histories = {int(key): value for key, value in self.histories.items()}
        self.is_first = True

    def flag(self, is_train):
        return 'train' if is_train else 'eval'

    def clear_history(self, epoch_idx):
        self.histories = {idx: history for idx, history in self.histories.items() if idx < epoch_idx}

    def forward(self, value, epoch_idx, is_train):
        if value is None:
            return
        if self.is_first:
            self.clear_history(epoch_idx)
            self.is_first = False
        if epoch_idx not in self.histories:
            self.histories[epoch_idx] = {}
        epoch_histories = self.histories[epoch_idx]
        if self.flag(is_train) not in epoch_histories:
            epoch_histories[self.flag(is_train)] = []
        epoch_histories[self.flag(is_train)].append(value)

    def step_avg(self, step2log, step_idx, epoch_idx, is_train):
        if epoch_idx not in self.histories:
            return None
        epoch_histories = self.histories[epoch_idx]
        if self.flag(is_train) not in epoch_histories:
            return None
        histories = epoch_histories[self.flag(is_train)]
        values = histories[step_idx-step2log:step_idx]
        return sum(values) / len(values)

    def epoch_avg(self, epoch_idx, is_train):
        if epoch_idx not in self.histories:
            return None
        epoch_histories = self.histories[epoch_idx]
        if self.flag(is_train) not in epoch_histories:
            return None
        values = self.histories[epoch_idx][self.flag(is_train)]
        return sum(values) / len(values)

    def check_exist(self, epoch_idx, is_train):
        return self.flag(is_train) in self.histories[epoch_idx]

    def to_file(self):
        json.dump(self.histories, self.json_file.open('w'))

    def epoch_indexes(self):
        return self.histories.keys()

    def epoch_avgs(self, is_train, epoch_idx=None):
        valid_idxes = self.histories.keys() if epoch_idx is None else [idx for idx in self.histories.keys() if idx <= epoch_idx]
        avgs = {idx: self.epoch_avg(idx, is_train) for idx in valid_idxes}
        return {idx: avg for idx, avg in avgs.items() if avg is not None}


class Logger(PtBase):
    def __init__(self, name: str, work_dir: str, step2log: int, splits: List[str], ):
        self.name = name
        self.work_dir = to_path(work_dir)
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)
        self.step2log = step2log
        self.metric = Metric(name, self.work_dir)
        self.splits = splits
        super().__init__()

    @property
    def controller(self):
        return self.controller

    def need_log(self, step_idx):
        if self.step2log == 0:
            return False
        return step_idx % self.step2log == 0 and step_idx != 0

    def flag(self, is_train):
        return 'train' if is_train else 'eval'

    def forward(self, value, step_idx, epoch_idx, is_train, need_log=True):
        self.metric.forward(value, epoch_idx, is_train)
        if need_log and self.need_log(step_idx):
            self.step_log(step_idx, epoch_idx, is_train)

    def epoch_update(self, epoch_idx, is_train):
        self.epoch_log(epoch_idx, is_train)
        self.metric.to_file()

    def epoch_log(self, epoch_idx, is_train):
        raise NotImplementedError

    def step_log(self, step_idx, epoch_idx, is_train):
        raise NotImplementedError

    @classmethod
    def build_loggers(cls, args, controller=None):
        args = to_namespace(args)
        loggers = {}
        for split in try_get_attr(args, f'{cls.prefix_name()}_splits', []):
            setattr(args, f'{cls.prefix_name()}_name', split)
            loggers[split] = cls.build(args, controller=controller)
        return loggers


class PrintLogger(Logger):
    step_printer = print

    def __init__(self,
                 name: str,
                 work_dir: str,
                 step2log: int,
                 step_log_format: str = 'Epoch: {}, Step: {}, {}',
                 epoch_log_format: str = 'Epoch: {}, {}',
                 ):
        super().__init__(name, work_dir, step2log)
        self.step_log_format = step_log_format
        self.epoch_log_format = epoch_log_format

    @classmethod
    def init_args(cls, params, sub_cls=None):
        cls.default_init_args(params)
        try_set_attr(params, f'{cls.prefix_name()}_work_dir',
                     to_path(params.root_dir).joinpath(f'loggers/{params.proj_name}'))

    def step_log(self, step_idx, epoch_idx, is_train):
        avg_value = self.metric.step_avg(self.step2log, step_idx, epoch_idx, is_train)
        str_avt = f'{avg_value:<8.4f}'
        message = f'''{self.flag(is_train)}_{self.name}: {colored(str_avt, 'green')}'''
        log = self.step_log_format.format(colored(epoch_idx, 'green'), colored(step_idx, 'green'), message)
        self.step_printer(log)

    def epoch_log(self, epoch_idx, is_train):
        avg_value = self.metric.epoch_avg(epoch_idx, is_train)
        str_avt = f'{avg_value:<8.4f}'
        message = f'''{self.flag(is_train)}_{self.name}: {colored(str_avt, 'green')}'''
        log = self.epoch_log_format.format(colored(epoch_idx, 'green'), message)
        print(log)

    @classmethod
    def install_step_printer(cls, printer):
        cls.step_printer = printer


class VisdomLogger(Logger):
    def __init__(self,
                 name: str,
                 work_dir: str,
                 step2log: int = 100,
                 server: str = 'http://219.223.169.76',
                 port: int = 8000,
                 env: str = None,
                 splits: List[str] = None,
                 ):
        super().__init__(name, work_dir, step2log, splits)
        self.visdom_env: VisdomEnv = VisdomEnv(server, port, env)
        self.windows = {
            self.window_name(True): self.visdom_env.new_line_win(self.window_name(True)),  # step window
            self.window_name(False): self.visdom_env.new_line_win(self.window_name(False))  # epoch window
        }
        self.first_flags = {'train': True, 'eval': True}

    @classmethod
    def init_args(cls, params, sub_cls=None):
        cls.default_init_args(params)
        try_set_attr(params, f'{cls.prefix_name()}_work_dir', to_path(params.root_dir).joinpath(f'loggers/{params.proj_name}'))
        try_set_attr(params, f'{cls.prefix_name()}_env', params.proj_name)

    def window_name(self, is_step):
        return f'''{self.name}_{'step' if is_step else 'epoch'}'''

    def step_log(self, step_idx, epoch_idx, is_train):
        avg_value = self.metric.step_avg(self.step2log, step_idx, epoch_idx, is_train)
        if avg_value is None:
            return
        window: LineWin = self.windows[self.window_name(True)]
        window_opt = {'title': self.name, 'xlabel': 'step'}
        x, y = np.asarray([step_idx]), np.asarray([avg_value])
        is_reset = step_idx == self.step2log
        window.plot(x, y, self.flag(is_train), window_opt, is_reset)

    def epoch_log(self, epoch_idx, is_train):
        if self.first_flags[self.flag(is_train)] and epoch_idx != 0:
            epoch_avgs = self.metric.epoch_avgs(is_train, epoch_idx)
            x, y = list(epoch_avgs.keys()), list(epoch_avgs.values())
        else:
            epoch_avg = self.metric.epoch_avg(epoch_idx, is_train)
            if epoch_avg is None:
                return
            x, y = [epoch_idx], [epoch_avg]
        if self.first_flags[self.flag(is_train)]:
            self.first_flags[self.flag(is_train)] = False
        x, y = np.asarray(x), np.asarray(y)
        window: LineWin = self.windows[self.window_name(False)]
        window_opt = {'title': self.name, 'xlabel': 'epoch'}
        is_reset = False
        window.plot(x, y, self.flag(is_train), window_opt, is_reset)

    def show_log(self):
        for is_train in (True, False):
            idxes = self.metric.epoch_indexes()
            if len(idxes) == 0:
                return
            epoch_avgs = [(idx, self.metric.epoch_avg(idx, is_train)) for idx in idxes if self.metric.check_exist(idx, is_train)]
            if len(epoch_avgs) == 0:
                return
            x, y = zip(*epoch_avgs)
            x, y = np.asarray(x), np.asarray(y)
            window: LineWin = self.windows[self.window_name(False)]
            window_opt = {'title': self.name, 'xlabel': 'epoch'}
            is_reset = False
            window.plot(x, y, self.flag(is_train), window_opt, is_reset)


class LoggerGroup(Logger):
    logger_group = None

    def __new__(cls, *args, **kwargs):
        if cls.logger_group is None:
            cls.logger_group = super(LoggerGroup, cls).__new__(cls)
        return cls.logger_group

    def __init__(self,
                 name: str,
                 work_dir: str,
                 step2log: int = 100,
                 logger_cls: str = 'print_logger',
                 logger_kwargs=None,
                 ):
        super().__init__(name, work_dir, step2log)
        self.logger_cls = LoggerGroup.load_cls(logger_cls)
        self.logger_kwargs = logger_kwargs
        self.loggers = {}

    @property
    def logger_names(self):
        return tuple(self.loggers.keys())

    @classmethod
    def add_args(cls, group, params=None, sub_cls=None):
        group = cls.default_add_args(group, params)
        logger_cls = cls.load_cls(try_get_attr(params, f'{cls.prefix_name()}_logger_cls', check=False))
        if logger_cls is not None:
            group = logger_cls.add_args(group, params)
        return group

    @classmethod
    def init_args(cls, params, sub_cls=None):
        cls.default_init_args(params)
        try_set_attr(params, f'{cls.prefix_name()}_name', 'logger_group')
        try_set_attr(params, f'{cls.prefix_name()}_logger_dir',
                     to_path(params.root_dir).joinpath(f'loggers/{params.proj_name}'))
        logger_cls = cls.load_cls(try_get_attr(params, f'{cls.prefix_name()}_logger_cls', check=False))
        if logger_cls is not None:
            logger_cls.init_args(params)
            setattr(params, f'{cls.prefix_name()}_logger_kwargs', load_func_kwargs(params, logger_cls.__init__, cls.prefix_name()))

    def forward(self, log, step_idx, epoch_idx, is_train, need_log=True):
        for name in self.logger_names:
            self.loggers[name].forward(log[name], step_idx, epoch_idx, is_train, need_log)

    def epoch_update(self, epoch_idx, is_train):
        for logger in self.loggers.values():
            logger.epoch_update(epoch_idx, is_train)

    def install_step_printer(self, printer):
        if self.logger_cls == PrintLogger:
            PrintLogger.install_step_printer(printer)

    def epoch_acc(self, epoch_idx, is_train):
        acc = None
        for logger_name in self.logger_names:
            if 'acc' in logger_name:
                acc = self.loggers[logger_name].metric.epoch_avg(epoch_idx, is_train)
                break
        if acc is None:
            sys_logger.info(f'Cant find acc in loggers names {self.logger_names}')
        return acc

    def register_logger(self, name):
        assert name not in self.loggers
        logger_kwargs = copy.deepcopy(self.logger_kwargs)
        logger_kwargs['logger_name'] = name
        if 'logger_dir' not in logger_kwargs:
            logger_kwargs['logger_logger_dir'] = self.work_dir
        if 'step2log' not in logger_kwargs:
            logger_kwargs['logger_step2log'] = self.step2log
        logger = self.logger_cls.build_modules(logger_kwargs)
        self.loggers[name] = logger
        return logger
    
    def __getitem__(self, item):
        return self.loggers[item]








