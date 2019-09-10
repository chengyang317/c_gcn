# coding=utf-8
import logging
from pt_pack.utils import to_seq
import threading

logger = logging.getLogger(__name__)

__all__ = ['FieldGroup', 'Field', 'ProxyField', 'SwitchField', "PseudoField"]


class Field(object):

    __slots__ = ('name', 'datas', 'idx_map_fn', 'data_iter')

    def __init__(self, name, datas=None):
        self.name = name
        self.datas = datas
        self.idx_map_fn = None
        self.data_iter = None

    def change_name(self, new_name):
        self.name = new_name
        return self

    def __getitem__(self, idx):
        if self.idx_map_fn is not None:
            idx = self.idx_map_fn(idx)
        return self.datas[idx]

    def __next__(self):
        data = next(self.data_iter)
        return data

    def __iter__(self):
        self.data_iter = iter(self.datas)
        return self

    def __len__(self):
        return len(self.datas)

    def append(self, item):
        self.datas.append(item)


class PseudoField(object):
    __slots__ = ('name', 'datas', 'idx_map_fn', 'data_iter')

    def __init__(self, name, datas=None, idx_map_fn=None):
        self.name = name
        self.datas = datas
        self.idx_map_fn = idx_map_fn

    def change_name(self, new_name):
        self.name = new_name
        return self

    def __getitem__(self, idx):
        if self.idx_map_fn is not None:
            idx = self.idx_map_fn(idx)
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


class SwitchField(object):
    def __init__(self, name, fields, switch_fn):
        self.name = name
        self.fields = fields
        self.switch_fn = switch_fn

    def __getitem__(self, idx):
        field = self.switch_fn(idx, self.fields)
        return field[idx]



class ProxyField(Field):
    def __init__(self, name, depends, depend_fn):
        super().__init__(name)
        self.depends = to_seq(depends)
        self.depend_fn = depend_fn
        self.cache = dict()
        # self.lock = threading.Lock()

    def __getitem__(self, idx):
        return self.depend_fn(*(depend[idx] for depend in self.depends))

    def __len__(self):
        return len(self.depends[0])

    def __next__(self):
        idx = next(self.data_iter)
        data = self.load_data(idx)
        return data

    def __iter__(self):
        self.data_iter = iter(range(len(self)))
        return self


class FieldGroup(object):
    def __init__(self, fields):
        self.fields = to_seq(fields)
        self._field_iters = None
        self.idx_map_fn = None

    @property
    def field_names(self):
        return (field.name for field in self.fields)

    @property
    def name2fields(self):
        return {field.name: field for field in self.fields}

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.idx_map_fn is not None:
                idx = self.idx_map_fn(idx)
            return [field[idx] for field in self.fields]
        return self.name2fields[idx]

    def __iter__(self):
        self._field_iters = [iter(field) for field in self.fields]
        return self

    def __next__(self):
        outs = []
        for field_iter in self._field_iters:
            outs.append(next(field_iter))
        return outs

    def __len__(self):
        if len(self.fields) > 0:
            return len(self.fields[0])
        return None

    def split(self):
        return self.fields