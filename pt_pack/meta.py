# coding=utf-8
import argparse
from pt_pack.utils import cls_name_trans, to_namespace, load_func_kwargs, add_func_args, init_func_args
from collections import defaultdict

__all__ = ['PtMeta', 'PtBase']


class PtMeta(type):
    cls_map = defaultdict(dict)
    # module_attr_names = ('add_args', 'default_add_args', 'load_cls', 'build', 'default_build', 'init_args',
    #                      'default_init_args', 'is_root_cls', 'root_cls', 'prefix_name')
    # common_attr_names = ('load_cls', 'is_root_cls', 'build', 'default_build', 'default_add_args', 'root_cls',
    #                      'prefix_name')

    def __new__(cls, cls_name, cls_bases, cls_attrs):
        # if not any((base_cls.__class__ is PtMeta for base_cls in cls_bases)):
        #     cls.fill_attrs(cls_attrs, cls.module_attr_names)
        # if cls_bases[0] is nn.Module:
        #     cls_attrs = cls.fill_attrs(cls_attrs, cls.module_attr_names)
        # elif cls_bases[0] is object or cls_name in ('PtDataset', 'PtOptimizer'):
        #     cls_attrs = cls.fill_attrs(cls_attrs, cls.common_attr_names)
        new_cls = super().__new__(cls, cls_name, cls_bases, cls_attrs)
        if 'cls_name' in cls_bases:
            cls_name = cls_bases['cls_name']
        cls.register_cls(cls_name, new_cls)
        return new_cls

    @classmethod
    def fill_attrs(cls, cls_attrs, attr_names):
        for name in attr_names:
            attr_value = getattr(cls, f'_{name}', None)
            if attr_value is None:
                raise ModuleNotFoundError(f'can not find attr _{name}')
            cls_attrs[name] = classmethod(attr_value)
        return cls_attrs

    @classmethod
    def load_cls(cls, sub_cls, sub_cls_name):
        if sub_cls_name is None:
            return None
        if any([char.isupper() for char in sub_cls_name]):
            sub_cls_name = cls_name_trans(sub_cls_name)
        if sub_cls.root_cls() is None:
            return cls.cls_map['PtBase'].get(sub_cls_name, None)
        return cls.cls_map[sub_cls.root_cls().__name__].get(sub_cls_name, None)

    @classmethod
    def register_cls(cls, sub_cls_name, sub_cls):
        sub_cls_name = cls_name_trans(sub_cls_name)
        root_cls = sub_cls.root_cls()
        if root_cls is None:
            assert sub_cls.__name__ == 'PtBase'
            root_cls = sub_cls
        root_cls_name = root_cls.__name__
        if sub_cls_name in cls.cls_map[root_cls_name]:
            raise KeyError(f'Class name {sub_cls_name} has been registered')
            # return
        cls.cls_map[root_cls_name][sub_cls_name] = sub_cls
        if sub_cls.is_root_cls():
            cls.cls_map['PtBase'][sub_cls_name] = sub_cls


class PtBase(object, metaclass=PtMeta):

    def __init__(self):
        self.controller = None

    @classmethod
    def default_add_args(cls, group: argparse.ArgumentParser, params):
        return add_func_args(group, cls.__init__, prefix=cls.prefix_name())

    @classmethod
    def prefix_name(cls):
        name = getattr(cls, 'prefix', cls.root_cls().__name__.lower())
        return name

    @classmethod
    def root_cls(cls):
        if cls.is_root_cls():
            return cls
        for base_cls in list(cls.__bases__):
            if base_cls.__class__ is PtMeta:
                return base_cls.root_cls()

    @classmethod
    def add_args(cls, group, params=None):
        cls.default_add_args(group, params)
        for p_cls in cls.__bases__:
            if hasattr(p_cls, 'add_args') and p_cls != PtBase:
                p_cls.add_args(group, params)
        return group

    @classmethod
    def load_cls(cls, sub_cls=None):
        if not isinstance(sub_cls, str):
            return sub_cls
        module_cls = PtMeta.load_cls(cls, sub_cls)
        return module_cls

    @classmethod
    def default_build(cls, params, controller=None):
        init_kwargs = load_func_kwargs(params, cls.__init__, cls.prefix_name())
        module = cls(**init_kwargs)
        if controller is not None:
            module.controller = controller
        # if controller is not None:
        #     controller.register_module(module)
        return module

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        params = to_namespace(params)
        if cls.is_root_cls():
            sub_cls = cls.load_cls(getattr(params, f'{cls.prefix_name()}_cls', sub_cls))
            if sub_cls is not None:
                if cls is sub_cls and 'build' not in cls.__dict__:
                    return cls.default_build(params, controller=controller)
                return sub_cls.build(params, controller=controller)
            else:
                return cls.default_build(params, controller=controller)
        else:
            return cls.default_build(params, controller=controller)

    @classmethod
    def init_args(cls, params):
        params = cls.default_init_args(params)
        for p_cls in cls.__bases__:
            if hasattr(p_cls, 'init_args') and p_cls != PtBase:
                p_cls.init_args(params)
        return params

    @classmethod
    def default_init_args(cls, params):
        return init_func_args(cls.__init__, params, cls.prefix_name())

    @classmethod
    def is_root_cls(cls):
        return any((base_cls.__name__ == 'PtBase' for base_cls in cls.__bases__))
        # return not any((base_cls.__class__ is PtMeta for base_cls in cls.__bases__))

    def init(self):
        """
        init function for each module
        :return:
        """
        pass

    def before_epoch_train(self):
        pass

    def step_train(self, message):
        return message

    def before_step_train(self, batch_idx):
        pass

    def after_step_train(self):
        pass

    def after_epoch_train(self):
        pass

    def before_epoch_eval(self):
        pass

    def before_step_eval(self, batch_idx):
        pass

    def step_eval(self, message):
        return message

    def after_step_eval(self):
        pass

    def after_epoch_eval(self):
        pass



