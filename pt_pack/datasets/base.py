# coding=utf-8
import torch.utils.data as py_data
from .reader import Reader
import logging
from pt_pack.utils import to_path, try_get_attr, load_func_params, add_argument, str_bool, add_func_args, load_func_kwargs
from pt_pack.meta import PtBase
from typing import List


sys_logger = logging.getLogger(__name__)


__all__ = ['Dataset']


class Dataset(py_data.Dataset, Reader, PtBase):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 req_field_names: List[str],
                 ):
        self.data_dir = to_path(data_dir)
        self.split = split
        self.loaders = None
        py_data.Dataset.__init__(self)
        Reader.__init__(self, req_field_names)
        PtBase.__init__(self)

    @classmethod
    def add_args(cls, group, params=None):
        super().add_args(group, params)
        add_argument(group, f'{cls.prefix_name()}_splits', type=str, nargs='*')
        add_argument(group, f'{cls.prefix_name()}_shuffles', type=str_bool, nargs='*')
        add_argument(group, f'{cls.prefix_name()}_belongs', type=str, nargs='*')
        add_func_args(group, py_data.DataLoader.__init__, cls.prefix_name())

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        loaders = {}
        splits = try_get_attr(params, f'{cls.prefix_name()}_splits', ('train',))
        shuffles = try_get_attr(params, f'{cls.prefix_name()}_shuffles', (True,))
        belongs = try_get_attr(params, f'{cls.prefix_name()}_belongs', ('train',))

        loader_kwargs = load_func_kwargs(params, py_data.DataLoader.__init__, cls.prefix_name())
        init_kwargs = load_func_kwargs(params, cls.__init__, cls.prefix_name())
        cls_name = getattr(params, f'{cls.prefix_name()}_cls', None)
        assert cls_name is not None
        sub_cls = cls.load_cls(cls_name)
        for idx, belong in enumerate(belongs):
            init_kwargs['split'] = splits[idx]
            dataset = sub_cls(**init_kwargs)
            if controller is not None:
                dataset.controller = controller
            loader_kwargs.update({
                'shuffle': shuffles[idx],
                'dataset': dataset,
                'collate_fn': dataset.collate_fn
            })
            loaders[belong] = py_data.DataLoader(**loader_kwargs)
        return loaders

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.name2fields[item]
        datas = [data for data in self.field_group[item]]
        return datas

    def collate_fn(self, batch):
        batch_dict = dict(zip(self.req_field_names, zip(*batch)))
        # assert not any([is_nan(var) for var in batch_dict.values()]), 'data should not be Nan'
        return batch_dict
































