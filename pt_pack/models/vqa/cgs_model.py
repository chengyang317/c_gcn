# coding=utf-8
from ..base import NetModel
import logging
from pt_pack.modules import Net
from pt_pack.utils import try_set_attr, try_get_attr, load_func_kwargs


sys_logger = logging.getLogger(__name__)


class CgsModel(NetModel):
    net_names = ('q_net', 'graph_net', 'cls_net')

    def __init__(self,
                 q_net: Net,
                 graph_net: Net,
                 cls_net: Net,
                 ):
        super().__init__({'q_net': q_net, 'graph_net': graph_net, 'cls_net': cls_net})
        self.q_net = q_net
        self.graph_net = graph_net
        self.cls_net = cls_net

    def forward(self, img_obj_feats, q_labels, q_lens):
        q_feat = self.q_net(q_labels, q_lens)
        img_feat = self.graph_net(img_obj_feats, q_feat)
        logit = self.cls_net(img_feat, q_feat)
        return logit

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_q_net_cls', 'cgs_q_net')
        try_set_attr(params, f'{cls.prefix_name()}_graph_net_cls', 'cgs_graph_net')
        try_set_attr(params, f'{cls.prefix_name()}_cls_net_cls', 'cgs_cls_net')
        for net_name in cls.net_names:
            net_cls = Net.load_cls(try_get_attr(params, f'{cls.prefix_name()}_{net_name}_cls', check=False))
            if net_cls is not None:
                net_cls.init_args(params)







































































































































