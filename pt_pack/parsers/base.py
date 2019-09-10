import argparse
import logging
from pt_pack.utils import to_namespace, to_seq, str_bool, add_argument, to_path
from typing import Dict
from pt_pack.meta import PtBase
import copy
import json
import os.path as osp


sys_logger = logging.getLogger(__name__)


__all__ = ['Parser']


PARSER_MAP = {}


def paser_register(parser_cls):
    global PARSER_MAP
    assert parser_cls.module_type not in PARSER_MAP
    PARSER_MAP[parser_cls.module_type] = parser_cls
    return parser_cls


class BaseParser(object):
    module_type = 'base'

    def __init__(self,
                 root_parser=None,
                 init_params=None
                 ):
        self.init_params = init_params
        self.root_parser, self.group_parser = self.add_parser(root_parser)
        self.dynamic_add_args()
        # self.params = self.parse()

    def add_parser(self, root_parser: argparse.ArgumentParser = None):
        if root_parser is None:
            root_parser = argparse.ArgumentParser(description=f'{self.module_type} root parser')
            group_parser = root_parser
        else:
            group_parser = root_parser.add_argument_group(f'{self.module_type} group parser')

        module_cls = self.init_params.get(f'{self.module_type}_cls', self.module_type)
        add_argument(group_parser, f'{self.module_type}_cls', type=str, help=f'{self.module_type} class name',
                                  default=module_cls)
        return root_parser, group_parser

    def parse(self):
        params = self.unknown_parse(self.init_params)  # start to recognize args of layer level
        params = self.init_args(params)  # start to init args from layer level
        return params

    def unknown_parse(self, kwargs):
        params = self.root_parser.parse_args()
        params = self.merge_from_kwargs(params, kwargs)
        return params

    def known_parse(self, kwargs):
        params, _ = self.root_parser.parse_known_args()
        params = self.merge_from_kwargs(params, kwargs)
        return params

    @property
    def root_cls(self):
        return PtBase.load_cls(self.module_type)

    def module_cls(self, params):
        return self.root_cls.load_cls(getattr(params, f'{self.module_type}_cls'))

    def dynamic_add_args(self):
        params = self.known_parse(self.init_params)
        assert getattr(params, f'{self.module_type}_cls', None) is not None
        self.add_args(params)

    def add_args(self, params):
        self.module_cls(params).add_args(self.group_parser, params)

    def init_args(self, params):
        self.module_cls(params).init_args(params)
        return params

    @staticmethod
    def merge_from_kwargs(params, kwargs):
        for key, value in kwargs.items():
            if key not in params:
                continue
            s_value = getattr(params, key, None)
            if s_value is None:
                setattr(params, key, value)
        return params


@paser_register
class ModelParser(BaseParser):
    module_type = 'model'

    def dynamic_add_args(self):
        params = self.known_parse(self.init_params)
        assert getattr(params, 'model_cls', None) is not None
        self.add_args(params)  # start to add args of model level
        params = self.known_parse(self.init_params)
        params = self.init_args(params)  # start recognizing the args of model level
        self.add_args(params)  # start to add args of net level
        # PtModel.add_args(self.groups['model'], args)
        params = self.known_parse(self.init_params)  # start to recognize args of net level
        params = self.init_args(params)  # start to init args from net level
        self.add_args(params)  # start to add args of layer level


@paser_register
class DatasetParser(BaseParser):
    module_type = 'dataset'


@paser_register
class CriterionParser(BaseParser):
    module_type = 'criterion'


@paser_register
class CudaParser(BaseParser):
    module_type = 'cuda'


@paser_register
class OptimizerParser(BaseParser):
    module_type = 'optimizer'

    def dynamic_add_args(self):
        super().dynamic_add_args()
        params = self.known_parse(self.init_params)
        params = self.init_args(params)
        self.add_args(params)


@paser_register
class CheckpointParser(BaseParser):
    module_type = 'checkpoint'


@paser_register
class ExperimentParser(BaseParser):
    module_type = 'experiment'


@paser_register
class TrainerParser(BaseParser):
    module_type = 'trainer'

    def add_parser(self, root_parser: argparse.ArgumentParser = None):
        if root_parser is None:
            root_parser = argparse.ArgumentParser(description=f'{self.module_type} root parser')
            group_parser = root_parser
        else:
            group_parser = root_parser.add_argument_group(f'{self.module_type} group parser')

        group_parser.add_argument(f'--{self.module_type}_cls', type=str, default='trainer',
                                  help=f'{self.module_type} class name')
        return root_parser, group_parser


@paser_register
class EnvParser(BaseParser):
    module_type = 'env'

    def dynamic_add_args(self):
        work_dir = self.init_params.get('work_dir', None) or './work_dir'
        proj_name = self.init_params.get('proj_name', None) or 'project'
        exp_name = self.init_params.get('exp_name', None) or 'experiment'
        exp_version = self.init_params.get('exp_version', None) or -1
        auto_load = self.init_params.get('auto_load', None) or False
        add_argument(self.group_parser, 'work_dir', type=str, default=work_dir, help='where to work')
        add_argument(self.group_parser, 'proj_name', type=str, default=proj_name, help=' name of project')
        add_argument(self.group_parser, 'exp_name', type=str, default=exp_name, help='experiment name')
        add_argument(self.group_parser, 'exp_version', default=exp_version, help='experiment version')
        add_argument(self.group_parser, 'config_file', type=str, help='config file path')
        add_argument(self.group_parser, 'auto_load', type=str_bool, default=auto_load, help='auto load config file')

    def parse(self):
        params = self.known_parse(self.init_params)  # start to recognize args of layer level
        return params


class Parser(PtBase):
    def __init__(self,
                 init_params: Dict,
                 module_types=('env', 'dataset', 'criterion', 'model', 'trainer', 'logger', 'checkpoint'),
                 **kwargs
                 ):
        super().__init__()
        self.init_params = init_params
        self.parser = argparse.ArgumentParser(description='Pt pack Toolkit', **kwargs)
        self.groups: Dict[str, BaseParser] = {}
        self.module_types = self._module_types_process(module_types)
        self.update_init_params()
        self.add_groups()
        self.params = None
        self.parse()

    def _module_types_process(self, module_types):
        module_types = list(module_types)
        if 'env' in module_types:
            module_types.remove('env')
        module_types.insert(0, 'env')
        return module_types

    def add_groups(self):
        for module_type in self.module_types:
            self.groups[module_type] = self.module_parser(module_type)

    def module_parser(self, module_type) -> BaseParser:
        if module_type not in self.groups:
            self.groups[module_type] = PARSER_MAP[module_type](self.parser, self.init_params)
        return self.groups[module_type]

    def auto_config_file(self, params):
        config_dir = to_path(f'./{params.work_dir}/{params.proj_name}/{params.exp_name}/{params.exp_version}/experiment')
        if not config_dir.exists():
            config_dir.mkdir(parents=True)
        return config_dir.joinpath('params.json')

    def update_init_params(self):
        if 'env' not in self.module_types:
            return
        env_parser = self.module_parser('env')
        env_params = env_parser.parse()
        params = {}
        config_file = None
        if env_params.auto_load and env_params.config_file is not None:
            config_file = self.auto_config_file(params) if self.auto_config_file(params).exists() else None
        config_file = config_file or env_params.config_file
        if config_file is not None:
            file_config = self.load_config(config_file)
            for key, value in file_config.items():
                params[key] = self.init_params.get(key, None) or value

        for key, value in self.init_params.items():
            if key not in params:
                params[key] = value
        self.init_params = params

    def load_config(self, config_file):
        config_file = to_path(config_file)
        assert config_file.exists()
        config = json.load(open(config_file))
        return config

    def parse(self):
        params_list = list()
        for module_type, group in self.groups.items():
            # print(f'Parsing group for module type {module_type}')
            params_list.append(group.parse())
        ret_params = {}
        for params in params_list:
            for param_name, param_value in vars(params).items():
                if param_name in ret_params and ret_params[param_name] is not None:
                    continue
                if callable(param_value):
                    continue
                ret_params[param_name] = param_value
        self.params = to_namespace(ret_params)

    def save(self):
        assert self.params is not None
        params = vars(self.params)
        tags = {}
        for key, value in params.items():
            if not callable(value):
                tags[key] = value
            else:
                print(f'{key} can not be saved')
        json.dump(tags, open(self.auto_config_file(self.params), 'w'))











# class Parser(object):
#
#     def __init__(self,
#                  parser=None,
#                  **kwargs):
#         self.parser = self.add_root_parser(parser)
#         self.groups = {}
#         self.args = self.parse(kwargs)
#         self.echo_args(self.args)
#
#     def echo_args(self, args):
#         if not args.verbose:
#             return
#         pprint.pprint(args)
#
#     def parse(self, kwargs):
#         args = self.known_parse(kwargs)
#         self.add_args(self.parser, args)  # start to add args of model level
#         args = self.known_parse(kwargs)
#         args = self.init_args(args)  # start recognizing the args of model level
#         self.add_args(self.parser, args)  # start to add args of net level
#         # PtModel.add_args(self.groups['model'], args)
#         args = self.unknown_parse(kwargs)  # start to recognize args of net level
#         args = self.init_args(args)  # start to init args from net level
#         self.add_args(self.parser, args)  # start to add args of layer level
#         args = self.unknown_parse(kwargs)  # start to recognize args of layer level
#         args = self.init_args(args)  # start to init args from layer level
#         self.save_to_file(args)
#         return args
#
#     def known_parse(self, kwargs):
#         args, _ = self.parser.parse_known_args()
#         args = self.merge_arg_from_kwargs(args, kwargs)
#         args = self.merge_arg_from_file(args)
#         return args
#
#     def unknown_parse(self, kwargs):
#         args = self.parser.parse_args()
#         args = self.merge_arg_from_kwargs(args, kwargs)
#         args = self.merge_arg_from_file(args)
#         return args
#
#     def add_args(self, parser, args):
#         module_names = ('model', 'dataset', 'checkpoint', 'logger', 'cuda', 'optimizer', 'criterion')
#         modules = (Model, Dataset, Checkpoint, Logger, Cuda, Optimizer, Criterion)
#         for name, module in zip(module_names, modules):
#             if getattr(args, f'{name}_cls', None) is not None:
#                 module = module.load_cls(getattr(args, f'{name}_cls'))
#             if name not in self.groups:
#                 group = parser.add_argument_group(f'{module.prefix_name()}-specific configuration')
#                 self.groups[name] = group
#                 module.add_args(group, args)
#             else:
#                 module.add_args(self.groups[name], args)
#         return parser
#
#     def init_args(self, args):
#         if args.proj_name is None:
#             args.proj_name = f'dataset:{get_cls_name(args.dataset_cls)}-model:{get_cls_name(args.model_cls)}'
#         if args.root_dir is None:
#             args.root_dir = './work_dir'
#
#         module_names = ('model', 'dataset', 'checkpoint', 'logger', 'cuda', 'optimizer', 'criterion')
#         modules = (Model, Dataset, Checkpoint, Logger, Cuda, Optimizer, Criterion)
#         for name, module in zip(module_names, modules):
#             if getattr(args, f'{name}_cls', None) is not None:
#                 module = module.load_cls(getattr(args, f'{name}_cls'))
#             module.init_args(args)
#         return args
#
#     @classmethod
#     def build(cls, **kwargs):
#         return cls(**kwargs)
#
#     def add_root_parser(self, parser=None):
#         if parser is None:
#             parser = argparse.ArgumentParser(description=f'Pt Pack Toolkit')
#         parser.add_argument('--proj_name', type=str, help='name of the project')
#         parser.add_argument('--verbose', type=bool)
#         parser.add_argument('--debug', type=bool, help='debug mode or not')
#         parser.add_argument('--cfg_file', type=str, help='config file to recover from')
#         parser.add_argument('--root_dir', type=str, help='root directory')
#         parser.add_argument('--seed', type=int, help='seed')
#         parser.add_argument('--dataset_cls', type=str, help='dataset class name')
#         parser.add_argument('--model_cls', type=str, help='model class name')
#         parser.add_argument('--criterion_cls', type=str, help='criterion class name')
#         parser.add_argument('--optimizer_cls', type=str, help='optimizer class name')
#         parser.add_argument('--logger_cls', type=str, help='logger class name')
#         return parser
#
#     @staticmethod
#     def merge_arg_from_kwargs(args, kwargs):
#         for key, value in kwargs.items():
#             if key not in args:
#                 continue
#             s_value = getattr(args, key, None)
#             if s_value is None:
#                 setattr(args, key, value)
#         return args
#
#     @staticmethod
#     def merge_arg_from_file(args):
#         if try_get_attr(args, 'cfg_file', None) is None:
#             return args
#         root_dir = to_path(try_get_attr(args, 'root_dir', './work_dir'))
#         config_dir = root_dir.joinpath('configs')
#         cfg_file = config_dir.joinpath(args.cfg_file)
#         if not cfg_file.name.endswith('.yaml'):
#             cfg_file = cfg_file.parent.joinpath(cfg_file.name + '.yaml')
#
#         if not cfg_file.exists():
#             sys_logger.info(f'cfg file {cfg_file} does not exists')
#             return args
#         yaml_data = yaml.load(cfg_file.open())
#         for key, value in yaml_data.items():
#             if key not in args:
#                 continue
#             s_value = try_get_attr(args, key, None)
#             if s_value is None:
#                 setattr(args, key, value)
#         return args
#
#     @staticmethod
#     def save_to_file(args):
#         if args.cfg_file is not None:
#             return
#         config_dir = to_path(args.root_dir).joinpath('configs')
#         if not config_dir.exists():
#             config_dir.mkdir(parents=True)
#         config_file = config_dir.joinpath(f'{args.proj_name}.yaml')
#         yaml.dump(vars(args), config_file.open('w'))





