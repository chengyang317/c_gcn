# coding=utf-8
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict


__all__ = ['to_seq', 'fields_to_dict', 'to_path', 'partial_sum', 'message_statistic', 'batch_messages_fusion',
           'get_idx', 'is_nan', 'to_numpy', 'to_tensor', 'can_tensorfy', 'to_cuda', 'str_split', 'one_in', 'not_in',
           ]


def str_split(x_str: str, sep='_'):
    elem_tuple = x_str.split(sep)
    return [int(elem) if elem.isdigit() else elem for elem in elem_tuple]


def to_seq(data, default=None):
    if isinstance(data, (list, tuple)):
        return data
    elif data is None:
        return default
    return [data]


def to_numpy(batch_data):
    if isinstance(batch_data, (list, tuple)):
        if isinstance(batch_data[0], torch.Tensor):
            return torch.stack(batch_data).cpu().numpy()
        elif isinstance(batch_data[0], np.ndarray):
            return np.stack(batch_data)
        else:
            batch_data = [np.asarray(dset_data) for dset_data in batch_data]
            return np.stack(batch_data)
    elif isinstance(batch_data, torch.Tensor):
        return batch_data.cpu().numpy()
    else:
        assert isinstance(batch_data, np.ndarray), f'dset_datas type is {type(batch_data)}'
        return batch_data


def can_tensorfy(data):
    if isinstance(data, str):
        return False
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return False
        return can_tensorfy(data[0])
    return True


def to_tensor(data):
    if not can_tensorfy(data) or torch.is_tensor(data):
        return data
    elif isinstance(data, (list, tuple)):
        if not torch.is_tensor(data[0]):
            data = [to_tensor(item) for item in data]
        return torch.stack(data, dim=0)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)


def fields_to_dict(fields):
    if isinstance(fields, dict) or fields is None:
        return fields
    fields = fields if isinstance(fields, (tuple, list)) else [fields]
    return {field.name: field for field in fields}


def to_path(file_path) -> Path:
    if file_path is None or isinstance(file_path, Path):
        return file_path
    return Path(file_path)


def partial_sum(iter_x, until_idx):
    items = (item for idx, item in enumerate(iter_x) if idx < until_idx)
    return sum(items)


def get_idx(sequence, key):
    return [idx for idx, name in enumerate(sequence) if key == name][0]


def is_nan(var):
    return True if torch.isnan(var).sum().item() > 0 else False


def to_cuda(maybe_tensor, device, non_blocking=False):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.cuda(device, non_blocking)
    elif isinstance(maybe_tensor, dict):
        return {
            key: to_cuda(value, device, non_blocking)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, (list, tuple)):
        if isinstance(maybe_tensor[0], str):
            return maybe_tensor
        return [to_cuda(x, device, non_blocking) for x in maybe_tensor]
    else:
        raise NotImplementedError()


def one_in(queries, keys):
    queries = to_seq(queries)
    conditons = [query in keys for query in queries]
    return True if sum(conditons) > 0 else False


def not_in(queries, keys):
    queries = to_seq(queries)
    conditons = [query in keys for query in queries]
    return False if sum(conditons) > 0 else True


def message_statistic(batch_messages, tag='mean'):
    ret_dict = defaultdict(list)
    for message in batch_messages:
        for attr_name, attr in message.items():
            if attr.get('tags', None) is not None and tag in attr['tags']:
                ret_dict[attr_name].append(attr['value'])
    return {name: np.array(values).mean() for name, values in ret_dict.items()}


def batch_messages_fusion(batch_messages):
    ret_dict = defaultdict(dict)
    for idx, message in enumerate(batch_messages):
        for name, attr in message.items():
            if idx == 0:
                ret_dict[name] = {'name': name, 'value': [attr['value']], 'tags': attr['tags']}
            else:
                ret_dict[name]['value'].append(attr['value'])
    for attr in ret_dict.values():
        if 'mean' in attr['tags']:
            attr['value'] = np.array(attr['value']).mean()
    return ret_dict















