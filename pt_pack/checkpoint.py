# coding=utf-8
import logging
import torch
import torch.nn as nn
from pt_pack.utils import try_set_attr, cls_name_trans, to_path
from pt_pack.meta import PtBase
import os.path as osp


sys_logger = logging.getLogger(__name__)

__all__ = ['Checkpoint']


class Checkpoint(PtBase):
    file_format = 'model_{model_name}_{epoch_idx}.pth'

    def __init__(self,
                 save_dir: str,
                 restore_file: str = None,
                 epoch2save: int = 1,
                 only_best: bool = True,
                 strict_mode: str = False,
                 monitor: str = 'val_loss',
                 ):
        super().__init__()
        self.save_dir = to_path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.restore_file = to_path(restore_file)
        self.epoch2save = epoch2save
        self.only_best = only_best
        self.strict_mode = strict_mode
        self.monitor = monitor
        self.best_val_acc = 0

    @classmethod
    def init_args(cls, params):
        save_dir = osp.join(params.work_dir, f'{params.proj_name}/{params.exp_name}/{params.exp_version}/checkpoint')
        try_set_attr(params, f'{cls.prefix_name()}_save_dir', save_dir)

    def auto_search_state_file(self, model_name):
        # file_paths = list(self.checkpoint_dir.glob(self.file_format.format(epoch_idx='*', model_name=model_name)))
        file_paths = list(self.save_dir.glob('*.pth'))
        if not file_paths:
            return None, -1
        file_paths = sorted(file_paths, key=self._file2idx)
        return file_paths[-1], self._file2idx(file_paths[-1])

    @staticmethod
    def _file2idx(file):
        try:
            base_name = file.parts[-1]
            return int(base_name[(base_name.rindex('_') + 1):-4])
        except:
            sys_logger.info('Can not get epoch idx from {file}')
            return -1

    def load_checkpoint(self, model: nn.Module, strict_mode=None):
        strict_mode = strict_mode or self.strict_mode
        model_name = model.__class__.__name__
        if self.restore_file is not None:
            restore_file = self.restore_file
            last_epoch = -1
        else:
            restore_file, last_epoch = self.auto_search_state_file(model_name)
        if restore_file is not None:
            sys_logger.info(f'Loading checkpoint file {restore_file}')
            if hasattr(model, 'load_checkpoint'):
                model.load_checkpoint(restore_file)
            else:
                state_dict = torch.load(restore_file, map_location='cpu')
                model.load_state_dict(state_dict, strict_mode)
        self._last_epoch = last_epoch
        return last_epoch

    @property
    def last_epoch(self):
        return getattr(self, '_last_epoch', -1)

    def delete_state_files(self, model_name):
        file_paths = list(self.save_dir.glob(self.file_format.format(epoch_idx='*', model_name=model_name)))
        for file in file_paths:
            file.unlink()

    def save_checkpoint(self, epoch_idx, model: nn.Module, val_acc=None):
        if not self.need_save(epoch_idx, val_acc):
            return
        if hasattr(model, 'save_checkpoint'):
            return model.save_checkpoint(epoch_idx, self.save_dir, val_acc)
        model_name = cls_name_trans(model.__class__.__name__)
        if self.only_best:
            self.delete_state_files(model_name)
        torch.save(model.state_dict(), self.save_dir.joinpath(self.file_format.format(epoch_idx=epoch_idx, model_name=model_name)))

    def need_save(self, epoch_idx, val_acc=None):
        if self.epoch2save == -1:
            return False
        condition = epoch_idx % self.epoch2save == 0
        if val_acc is None:
            return condition
        if self.only_best:
            condition = condition and (val_acc is not None) and val_acc > self.best_val_acc
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        return condition

    def init(self):
        self.load_checkpoint(self.controller.real_model)

    def after_epoch_train(self):
        trainer = self.controller.trainer
        try:
            val_acc = trainer.running_messages['eval'][trainer.epoch_idx]['acc']['value']
        except:
            val_acc = 0
        self.save_checkpoint(trainer.epoch_idx, trainer.real_model, val_acc)






