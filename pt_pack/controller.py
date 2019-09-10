from pt_pack.meta import PtBase
from pt_pack.parsers import Parser
import copy
from pt_pack.utils import to_namespace
from pt_pack.models.base import Model
import torch.nn as nn


class Controller(object):
    def __init__(self,
                 init_params=None,
                 module_types=('parser', 'trainer', 'model', 'criterion', 'checkpoint', 'cuda', 'optimizer',
                               'dataset', 'experiment')
                 ):
        self.init_params = init_params or {}
        self.module_types = self._process_module_types(module_types)
        self.module_dicts = {}
        self.params = None
        self.build_modules()

    def _process_module_types(self, module_types):
        module_types = list(module_types)
        if 'checkpoint' in module_types:
            module_types.remove('checkpoint')
            module_types.insert(0, 'checkpoint')
        return module_types

    def build_modules(self):
        module_types = copy.deepcopy(self.module_types)
        params = None
        if 'parser' in module_types:
            module_types.remove('parser')
            parser = Parser(self.init_params, module_types)
            self.module_dicts['parser'] = parser
            params = parser.params
            parser.save()
        self.params = params or to_namespace(self.init_params)

        for module_type in module_types:
            # print(f'Building module {module_type}')
            module_root_cls = PtBase.load_cls(module_type)
            module = module_root_cls.build(self.params, controller=self)
            if module_type != 'dataset':
                self.register_module(module)
            else:
                self.module_dicts['dataset'] = module  # loaders

    def register_module(self, module: PtBase):
        module.controller = self
        self.module_dicts[module.prefix_name()] = module

    def train(self):
        assert 'trainer' in self.module_dicts
        return self.module_dicts['trainer']()

    def __getattr__(self, item):
        if item == 'loaders':
            return self.module_dicts['dataset']
        elif item == 'dataset':
            return list(self.loaders.values())[0].dataset
        elif item in self.module_types:
            return self.module_dicts.get(item, None)
        else:
            raise LookupError(f'no attr {item}')

    @property
    def real_model(self) -> Model:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

