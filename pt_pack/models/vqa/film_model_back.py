# coding=utf-8
from ..base import Model
from pt_pack.modules import Net
import argparse
from pt_pack.utils import try_get_attr, try_set_attr, str_bool
import torch

__all__ = ['Film']


class Film(Model):
    net_names = ('q', 'img', 'fusion', 'classifier')

    def __init__(self, q_net, img_net, fusion_net, classifier_net, net_modes=None, net_is_restores=None):
        super().__init__()
        self.q_net = q_net
        self.img_net = img_net
        self.fusion_net = fusion_net
        self.classifier_net = classifier_net
        self.net_modes = net_modes
        self.nets = (q_net, img_net, fusion_net, classifier_net)
        self.net_is_restores = net_is_restores

    def set_mode(self, is_train=None):
        if self.net_modes is not None and is_train is None:
            for net_mode, net in zip(self.net_modes, self.nets):
                net.train(net_mode)
        else:
            super().set_mode(is_train)

    def build_optimizers(self, optim_cls, optim_args):
        optimizers = {}
        for net_mode, net, net_name in zip(self.net_modes, self.nets, self.net_names):
            if not net_mode:
                continue
            params = filter(lambda p: p.requires_grad, net.parameters())
            optimizer = optim_cls(params, **optim_args)
            optimizers[net_name] = optimizer
        return optimizers

    def load_state_dict(self, state_dict, strict=True):
        new_dict = dict()
        for key, value in state_dict.items():
            for net_name, net_is_restore in zip(self.net_names, self.net_is_restores):
                if net_is_restore and net_name in key:
                    new_dict[key] = value
        super().load_state_dict(new_dict, strict)

    @classmethod
    def build(cls, params, model_cls_name=None, model_cls=None):
        cls_name_keys = [f'{name}_net_cls_name' for name in cls.net_names]
        nets = [Net.build(params, net_cls_name=try_get_attr(params, cls_name_key)) for cls_name_key in cls_name_keys]
        return cls(*nets, net_modes=params.net_modes, net_is_restores=params.net_is_restores)

    def forward(self, img, q_labels, q_lengths):
        if self.net_modes[0]:
            q_info = self.q_net(q_labels, q_lengths)
        else:
            with torch.no_grad():
                q_info = self.q_net(q_labels, q_lengths)
        if self.img_net is not None:
            if self.net_modes[1]:
                img_feat = self.img_net(img)
            else:
                with torch.no_grad():
                    img_feat = self.img_net(img)
        else:
            img_feat = img
        feat = self.fusion_net(img_feat, q_info)
        logit = self.classifier_net(feat)
        return logit

    @classmethod
    def get_input(cls, sample):
        img_key = 'images' if 'images' in sample else 'image_features_h5'
        ret_sample = [sample[key] for key in (img_key, 'question_labels', 'question_lengths')]
        return ret_sample

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, model_cls_name=None, model_cls=None):
        if not cls.has_add_args:
            group.add_argument('--q_net_cls_name', type=str)
            group.add_argument('--img_net_cls_name', type=str)
            group.add_argument('--fusion_net_cls_name', type=str)
            group.add_argument('--classifier_net_cls_name', type=str)
            group.add_argument('--net_modes',  nargs='*', type=str_bool)
            group.add_argument('--net_is_restores', nargs='*', type=str_bool)
        else:
            for net_name in ('q', 'img', 'fusion', 'classifier'):
                Net.add_args(group, params, try_get_attr(params, f'{net_name}_net_cls_name'))

    @classmethod
    def init_args(cls, params, model_cls_name=None, model_cls=None):
        try_set_attr(params, 'q_net_cls_name', 'film_q_net')
        try_set_attr(params, 'img_net_cls_name', 'film_img_net')
        try_set_attr(params, 'fusion_net_cls_name', 'film_fusion_net')
        try_set_attr(params, 'classifier_net_cls_name', 'film_classifier_net')
        try_set_attr(params, 'net_modes', (True, True, True, True))
        try_set_attr(params, 'net_is_restores', (True, True, True, True))
        for net_name in ('q', 'img', 'fusion', 'classifier'):
            Net.init_args(params, getattr(params, f'{net_name}_net_cls_name', None))
        return params


class AttnFilm(Film):
    def __init__(self, q_net, img_net, fusion_net, classifier_net):
        super().__init__(q_net, img_net, fusion_net, classifier_net)

    def forward(self, img, q_labels, q_lengths, fix_img_net=False):
        if self.net_modes[0]:
            q_info, q_query = self.q_net(q_labels, q_lengths)
        else:
            with torch.no_grad():
                q_info, q_query = self.q_net(q_labels, q_lengths)
        if self.img_net is not None:
            if self.net_modes[1]:
                img_feat = self.img_net(img)
            else:
                with torch.no_grad():
                    img_feat = self.img_net(img)
        else:
            img_feat = img
        feat = self.fusion_net(img_feat, q_info)
        logit = self.classifier_net(feat, q_query)
        return logit

    @classmethod
    def init_args(cls, params, model_cls_name=None, model_cls=None):
        try_set_attr(params, 'q_net_cls_name', 'attn_film_q_net')
        try_set_attr(params, 'img_net_cls_name', 'film_img_net')
        try_set_attr(params, 'fusion_net_cls_name', 'film_fusion_net')
        try_set_attr(params, 'classifier_net_cls_name', 'concat_classifier_net')
        try_set_attr(params, 'net_modes', (True, True, True, True))
        for net_name in ('q', 'img', 'fusion', 'classifier'):
            Net.init_args(params, getattr(params, f'{net_name}_net_cls_name', None))
        return params












