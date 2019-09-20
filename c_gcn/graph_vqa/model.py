# coding=utf-8
from pt_pack import Net, NetModel, try_get_attr, try_set_attr


class GraphVqaModel(NetModel):
    net_names = ('q_net', 'graph_net')

    def __init__(self,
                 q_net: Net,
                 graph_net: Net,
                 ):
        super().__init__({'q_net': q_net, 'graph_net': graph_net})
        self.q_net = q_net
        self.graph_net = graph_net

    def forward(self, img_obj_feats, q_labels, q_lens, q_ids, img_obj_nums=None, img_obj_boxes=None):
        q_feats = self.q_net(q_labels, q_lens)
        if img_obj_boxes is None:
            obj_feats, obj_boxes = img_obj_feats.split([img_obj_feats.shape[-1]-4, 4], dim=-1)
        else:
            obj_feats, obj_boxes = img_obj_feats, img_obj_boxes
        logits = self.graph_net(obj_feats, obj_boxes, q_feats, q_ids, img_obj_nums)
        return {'logits': {'name': 'logits', 'value': logits, 'tags': ('no_cpu',)}}

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_q_net_cls', 'cgs_graph_q_net')
        try_set_attr(params, f'{cls.prefix_name()}_graph_net_cls', 'cond_graph_vqa_net')
        for net_name in cls.net_names:
            net_cls = Net.load_cls(try_get_attr(params, f'{cls.prefix_name()}_{net_name}_cls', check=False))
            if net_cls is not None:
                net_cls.init_args(params)




