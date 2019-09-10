# coding=utf-8
from ..base import NetModel
import logging
from pt_pack.modules import Net
from pt_pack.utils import try_get_attr, try_set_attr
import torch


sys_logger = logging.getLogger(__name__)


__all__ = ['FilmModel']


class FilmModel(NetModel):
    net_names = ('q_net', 'img_net', 'fusion_net', 'cls_net')

    def __init__(self,
                 q_net: Net,
                 img_net: Net,
                 fusion_net: Net,
                 cls_net: Net,
                 ):
        super().__init__({'q_net': q_net, 'img_net': img_net, 'fusion_net': fusion_net, 'cls_net': cls_net})
        self.q_net = q_net
        self.img_net = img_net
        self.fusion_net = fusion_net
        self.cls_net = cls_net

    def forward(self, images, q_labels, q_lens):
        q_feats = self.q_net(q_labels, q_lens)
        img_feats = self.img_net(images)
        feats = self.fusion_net(img_feats, q_feats)
        logits = self.cls_net(feats, q_feats)
        return logits

    @classmethod
    def get_input(cls, sample):
        ret_sample = [sample[key] for key in ('images', 'q_labels', 'q_lens')]
        return ret_sample

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_q_net_cls', 'film_q_net')
        try_set_attr(params, f'{cls.prefix_name()}_img_net_cls', 'film_img_net')
        try_set_attr(params, f'{cls.prefix_name()}_fusion_net_cls', 'film_fusion_net')
        try_set_attr(params, f'{cls.prefix_name()}_cls_net_cls', 'film_cls_net')
        for net_name in cls.net_names:
            net_cls = Net.load_cls(try_get_attr(params, f'{cls.prefix_name()}_{net_name}_cls', check=False))
            if net_cls is not None:
                net_cls.init_args(params)

    def save_checkpoint(self, epoch_idx, cp_dir, val_acc):
        cp_dict = {
            'epoch_idx': epoch_idx,
            'val_acc': val_acc
        }
        for net_name in self.net_names:
            cp_dict.update({net_name: getattr(self, net_name).state_dict()})

        torch.save(cp_dict, cp_dir.joinpath(f'model_{epoch_idx}.pth'))

    def load_checkpoint(self, cp_file):
        state_dict = torch.load(cp_file, map_location='cpu')
        if 'q_net' not in state_dict:
            return self.load_state_dict(state_dict)
        else:
            for net_name in self.net_names:
                getattr(self, net_name).load_state_dict(state_dict[net_name])


















