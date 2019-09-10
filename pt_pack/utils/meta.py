from typing import Dict
import inspect
import argparse
from .option import add_argument
from typing import List


__all__ = ['to_namespace', 'try_get_attr', 'try_set_attr', 'cls_name_trans', 'get_cls_name',
           'load_func_kwargs', 'func_params', 'load_func_params', 'str_bool', 'arg_type_fix',
           'add_func_args', 'init_func_args']


def str_bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def to_namespace(args):
    if args is None:
        args = argparse.Namespace()
    elif isinstance(args, dict):
        return argparse.Namespace(**args)
    return args


def try_set_attr(args, key, value, check=True):
    if key not in args and check:
        return
    if getattr(args, key, None) is None:
        setattr(args, key, value)


def try_get_attr(args, key, default_value=None, check=True):
    if args is None:
        return default_value
    if not hasattr(args, key) and check:
        raise KeyError(f'{key} does not exist')
    value = getattr(args, key, None)
    return value if value is not None else default_value


def cls_name_trans(cls_name):
    def split_name(name):
        name_parts = []
        start_idx = 0
        for idx, char in enumerate(name):
            if idx == 0:
                continue
            if char.isupper():
                name_parts.append(name[start_idx:idx].lower())
                start_idx = idx
            else:
                continue
        name_parts.append(name[start_idx:].lower())
        return name_parts

    cls_name = '_'.join(split_name(cls_name))
    return cls_name


def get_cls_name(sub_cls):
    if sub_cls is None or isinstance(sub_cls, str):
        return sub_cls
    return sub_cls.__name__


def func_params(func):
    return inspect.signature(func).parameters


def load_func_kwargs(params, func, prefix=None) -> Dict:
    params = to_namespace(params)
    kwargs = {}
    prefix = f'{prefix}_' if prefix is not None else ''
    for parameter in func_params(func).values():
        p_name = parameter.name
        if p_name == 'self':
            continue
        defaut_value = parameter.default
        defaut_value = defaut_value if defaut_value is not parameter.empty else None
        kwargs[p_name] = try_get_attr(params, f'{prefix}{p_name}', defaut_value, check=False)
    return kwargs


def load_func_params(params, func, prefix=None) -> Dict:
    params = to_namespace(params)
    kwargs = {}
    prefix = f'{prefix}_' if prefix is not None else ''
    for parameter in func_params(func).values():
        p_name = parameter.name
        if p_name == 'self':
            continue
        if try_get_attr(params, f'{prefix}{p_name}', None, check=False) is not None:
            kwargs[f'{prefix}{p_name}'] = getattr(params, f'{prefix}{p_name}')
    return kwargs


def arg_type_fix(arg_type):
    if arg_type in [int, str, bool, float]:
        return str_bool if arg_type is bool else arg_type
    return arg_type


def add_func_args(group: argparse.ArgumentParser, func, prefix=''):
    parameters = func_params(func)
    if prefix != '':
        prefix += '_'

    for parameter in parameters.values():
        # if parameter.annotation is parameter.empty or parameter.name == 'self':
        if parameter.name == 'self':
            continue
        arg_type = parameter.annotation
        arg_name = parameter.name

        if arg_type is parameter.empty:
            add_argument(group, f'{prefix}{arg_name}')
        elif getattr(arg_type, '__origin__', None) is not None and arg_type.__origin__ in (list, List):
            add_argument(group, f'{prefix}{arg_name}', type=arg_type_fix(arg_type.__args__[0]), nargs='*')
        elif arg_type in [int, str, bool, float]:
            arg_type = arg_type_fix(arg_type)
            add_argument(group, f'{prefix}{arg_name}', type=arg_type)
    return group


def init_func_args(func, params, prefix=''):
    parameters = func_params(func)
    if prefix != '':
        prefix += '_'
    for parameter in parameters.values():
        if parameter.name == 'self':
            continue
        arg_name = parameter.name
        if parameter.default is not parameter.empty:
            try_set_attr(params, f'{prefix}{arg_name}', parameter.default)
    return params
