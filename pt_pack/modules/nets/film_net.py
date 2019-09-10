# coding=utf-8
import torch.nn as nn
from pt_pack.modules import Flatten, Conv2D, Linear, Null, Layer, SoftAttnLayer
import logging
from pt_pack.utils import str_bool, try_set_attr, partial_sum
from .base import Net
import argparse
import torch

sys_loger = logging.getLogger(__name__)

__all__ = ['FilmQNet', 'FilmImgNet', 'FilmClassifierNet', 'FilmFusionNet', 'DsodImgNet']


def is_odd(x):
    return x&1


def policy(flag, is_finetune=False):

    def fn(epoch):
        if is_odd(epoch//10):
            if is_odd(epoch//5):
                return not flag
            else:
                return flag
        else:
            if epoch < 10 and is_finetune:
                return False
            else:
                return True
    return fn


finetune_policy = {
    'img': policy(flag=False, is_finetune=True),
    'q': policy(False),
    'fusion': policy(True),
    'classifier': policy(True)
}


scratch_policy = {
    'img': policy(False),
    'q': policy(False),
    'fusion': policy(True),
    'classifier': policy(True),
}



class RnQNet(Net):
    def __init__(self, vocab_num, embed_dim, feat_dim, is_bi=False):
        super().__init__()
        self.embed_l = nn.Embedding(vocab_num, embed_dim)
        self.encoder_l = nn.GRU(embed_dim, feat_dim if not is_bi else feat_dim//2, batch_first=True, bidirectional=is_bi)
        self.reset_parameters()

    def forward(self, q_labels, q_lengths):
        q_embed = self.embed_l(q_labels)
        self.encoder_l.flatten_parameters()
        seq_out, _ = self.encoder_l(q_embed)
        q_feat = seq_out.gather(1, (q_lengths - 1).view(-1, 1, 1).expand(-1, -1, seq_out.size(-1))).squeeze()
        return q_feat

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--question_vocab_num', type=int)
        group.add_argument('--question_embed_dim', type=int)
        group.add_argument('--question_feat_dim', type=int)
        group.add_argument('--question_is_bi', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'question_vocab_num', 90)
        try_set_attr(params, 'question_embed_dim', 32)
        try_set_attr(params, 'question_feat_dim', 128)
        try_set_attr(params, 'question_is_bi', False)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='question')




class FilmQNet(Net):
    def __init__(self, vocab_num, embed_dim, feat_dim, info_dim, add_one, is_bi=False):
        super().__init__()
        self.info_dim = info_dim
        self.add_one = add_one
        self.embed_l = nn.Embedding(vocab_num, embed_dim)
        self.encoder_l = nn.GRU(embed_dim, feat_dim if not is_bi else feat_dim//2, batch_first=True, bidirectional=is_bi)
        self.linear_l = nn.Linear(feat_dim, info_dim)
        self.reset_parameters()

    def modify_q_info(self, q_info):
        q_info[:, 0, :] = q_info[:, 0, :] + 1.0
        return q_info

    def forward(self, q_labels, q_lengths):
        q_embed = self.embed_l(q_labels)
        self.encoder_l.flatten_parameters()
        seq_out, _ = self.encoder_l(q_embed)
        q_feat = seq_out.gather(1, (q_lengths - 1).view(-1, 1, 1).expand(-1, -1, seq_out.size(-1))).squeeze()
        q_info = self.linear_l(q_feat)
        q_info = q_info.view(q_info.shape[0], 2, -1)
        if self.add_one:
            q_info = self.modify_q_info(q_info)
        return q_info

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--question_vocab_num', type=int)
        group.add_argument('--question_embed_dim', type=int)
        group.add_argument('--question_feat_dim', type=int)
        group.add_argument('--question_info_dim', type=int)
        group.add_argument('--question_add_one', type=str_bool)
        group.add_argument('--question_is_bi', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'question_vocab_num', 90)
        try_set_attr(params, 'question_embed_dim', 200)
        try_set_attr(params, 'question_feat_dim', 4096)
        try_set_attr(params, 'question_add_one', True)
        try_set_attr(params, 'question_info_dim', 1024)
        try_set_attr(params, 'question_is_bi', False)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='question')


class AttnFilmQNet(FilmQNet):
    def __init__(self, vocab_num, embed_dim, feat_dim, info_dim, add_one, query_dim, is_bi=False):
        super().__init__(vocab_num, embed_dim, feat_dim, info_dim, add_one, is_bi)
        self.query_linear_l = nn.Linear(feat_dim, query_dim)
        self.reset_parameters()

    def forward(self, q_labels, q_lengths):
        q_embed = self.embed_l(q_labels)
        seq_out, _ = self.encoder_l(q_embed)
        q_feat = seq_out.gather(1, (q_lengths - 1).view(-1, 1, 1).expand(-1, -1, seq_out.size(-1))).squeeze()
        q_info = self.linear_l(q_feat)
        q_info = q_info.view(q_info.shape[0], 2, -1)
        if self.add_one:
            q_info = self.modify_q_info(q_info)
        q_query = self.query_linear_l(q_feat)
        return q_info, q_query

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--question_query_dim', type=int)
        super().add_args(group, params, cls_name, sub_cls)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'question_query_dim', 128)
        super().init_args(params, cls_name, sub_cls)


class ResImgNet(Net):
    def __init__(self):
        super().__init__()
        import torchvision
        resnet = torchvision.models.resnet101(False)
        layers = [
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            Conv2D(1024, 128, 3, padding=1, orders=('coord', 'conv', 'norm', 'act'))
        ]
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, img):
        img_feat = self.layers(img)
        return img_feat


class FilmImgNet(Net):
    default_orders = ('coord', 'conv', 'norm', 'act')

    def __init__(self, in_dim, layer_dims, layer_norm_types, layer_act_types, layer_coord_types,
                 layer_se_types, layer_orders=None):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_orders = layer_orders if layer_orders is not None else (self.default_orders,) * len(layer_dims)
        for idx, out_dim in enumerate(layer_dims):
            se_type = layer_se_types[idx]
            orders = layer_orders[idx]
            if se_type is not None:
                orders = list(orders)
                orders.insert(orders.index('act'), 'se')
            layer = Conv2D(in_dim, out_dim, 3, padding=1, orders=orders, norm_type=layer_norm_types[idx],
                           act_type=layer_act_types[idx], coord_type=layer_coord_types[idx], se_type=se_type)
            self.layers.append(layer)
            in_dim = out_dim
        self.reset_parameters()

    def forward(self, img_feat):
        layer_in = img_feat
        for layer in self.layers:
            layer_in = layer(layer_in)
        return layer_in

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--img_in_dim', type=int)
        group.add_argument('--img_layer_dims', nargs='*', type=int)
        group.add_argument('--img_layer_norm_types', nargs='*', type=str)
        group.add_argument('--img_layer_act_types', nargs='*', type=str)
        group.add_argument('--img_layer_coord_types', nargs='*', type=str)
        group.add_argument('--img_layer_se_types', nargs='*', type=str)


    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'img_in_dim', 1024)
        try_set_attr(params, 'img_layer_dims', (128,))
        if hasattr(params, 'img_layer_dims'):
            layer_num = len(params.img_layer_dims)
            try_set_attr(params, 'img_layer_norm_types', ('batch',) * layer_num)
            try_set_attr(params, 'img_layer_act_types', ('relu',) * layer_num)
            try_set_attr(params, 'img_layer_coord_types', ('default',) * layer_num)
            try_set_attr(params, 'img_layer_se_types', (None,) * layer_num)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='img')


class RnImgNet(Net):
    def __init__(self,):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2D(3, 32, 3, 2, 1),
            Conv2D(32, 64, 3, 2, 1),
            Conv2D(64, 128, 3, 2, 1),
            Conv2D(128, 128, 3, 2, 1)
        )
        self.reset_parameters()

    def forward(self, img):
        return self.layers(img)

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        pass

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        pass

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='img')


class PyramidFusionNet(Net):
    def __init__(self,
                 layer_types,
                 layer_dims,
                 layer_hs,
                 ):
        super().__init__()
        self.layer_dims = layer_dims
        self.layers = nn.ModuleList()
        from .layers.film_layers import PyramidFusionLayer
        for idx in range(len(layer_types)):
            layer_type = layer_types[idx]
            if layer_type is None:
                self.layers.append(None)
            else:
                self.layers.append(PyramidFusionLayer(in_dim=layer_dims[idx], feat_h=layer_hs[idx], layer_type=layer_types[idx]))
        self.reset_parameters()

    def forward(self, img_feats, q_info):
        gammas, betas = map(lambda x: x.squeeze(), q_info.chunk(2, dim=1))
        block_num = len([layer_dim for layer_dim in self.layer_dims if layer_dim is not None])
        gammas, betas = gammas.chunk(block_num, dim=1), betas.chunk(block_num, dim=1)
        feats = []
        true_idx = -1
        for idx, layer in enumerate(self.layers):
            layer_feat = img_feats[idx]
            if layer is None:
                continue
            true_idx += 1
            layer_feat = layer(layer_feat, gammas[true_idx], betas[true_idx])
            feats.append(layer_feat)
        return sum(feats)

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--fusion_layer_types', nargs='*', type=str)
        group.add_argument('--fusion_layer_dims', nargs='*', type=int)
        group.add_argument('--fusion_layer_hs', nargs='*', type=int)
        # group.add_argument('--fusion_layer_norm_types', nargs='*', type=str)
        # group.add_argument('--fusion_layer_act_types', nargs='*', type=str)
        # group.add_argument('--fusion_layer_fusion_norm_types', nargs='*', type=str)
        # group.add_argument('--fusion_layer_coords', nargs='*', type=str_bool)
        # group.add_argument('--fusion_layer_coord_types', nargs='*', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'fusion_layer_types', (None, 'film_fusion_layer', 'film_fusion_layer_f'))
        try_set_attr(params, 'fusion_layer_dims', (None, 128, 128))
        try_set_attr(params, 'fusion_layer_hs', (112, 56, 28))
        # if hasattr(args, 'fusion_layer_types'):
        #     layer_num = len(args.fusion_layer_types)
        #     try_set_attr(args, 'fusion_layer_norm_types', ('batch',)*layer_num)
        #     try_set_attr(args, 'fusion_layer_act_types', ('relu',) * layer_num)
        #     try_set_attr(args, 'fusion_layer_fusion_norm_types', ('instance',)*layer_num)
        #     try_set_attr(args, 'fusion_layer_coords', (True,) * layer_num)
        #     try_set_attr(args, 'fusion_layer_coord_types', ('default',)*layer_num)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='fusion')


class PyramidClassifierNet(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            Linear(in_dim, proj_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(proj_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, feat):
        logit = self.layers(feat)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_in_dim', type=int)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_in_dim', 128)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')



class DsodFusionNet(Net):
    def __init__(self,
                 in_dim,
                 layer_dims,
                 layer_types,
                 layer_feat_hs,
                 ):
        super().__init__()
        self.layer_types = layer_types
        self.layers = nn.ModuleList()
        self.layer_types = layer_types
        for idx in range(len(layer_types)):
            sub_args = argparse.Namespace(in_dim=in_dim, out_dim=layer_dims[idx], feat_h=layer_feat_hs[idx],)
            layer = Layer.build(sub_args, cls_name=layer_types[idx])
            in_dim = layer_dims[idx]
            self.layers.append(layer)
        self.layer_dims = layer_dims
        self._out_dim = in_dim
        self.reset_parameters()

    def forward(self, img, q_info):
        gammas, betas = map(lambda x: x.squeeze(), q_info.chunk(2, dim=1))
        in_feat = self.layers[0](img)
        infors = list()
        for layer_idx, layer in enumerate(self.layers[1:]):
            info_slice = slice(*(partial_sum(self.layer_dims[1:], layer_idx + i) for i in range(2)))
            in_feat, infor = layer(in_feat, gammas[:, info_slice], betas[:, info_slice])
            infors.append(infor)
        return infors

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--img_in_dim', type=int)
        group.add_argument('--img_layer_feat_hs', nargs='*', type=int)
        group.add_argument('--img_layer_dims', nargs='*', type=int)
        group.add_argument('--img_layer_types', nargs='*', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'img_in_dim', 3)
        try_set_attr(params, 'img_layer_types', ('img_init_layer', 'dsod_fusion_layer_a', 'dsod_fusion_layer_a', 'dsod_fusion_layer_a'))
        try_set_attr(params, 'img_layer_dims', (64, 128, 128, 128))
        try_set_attr(params, 'img_layer_feat_hs', (112, 56, 28, 14))
        return params

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='img')


class DsodFusionClassifierNet(Net):
    def __init__(self, answer_num, in_dim, fc_dim, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            Linear(in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, infors):
        infor = torch.cat(infors, dim=1)
        logit = self.layers(infor)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_in_dim', type=int)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_in_dim', 128 * 3)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class DsodFusionClassifierNetA(Net):
    def __init__(self, answer_num, in_dim, fc_dim, norm_type):
        super().__init__()
        self.layers = nn.Sequential(
            Linear(in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, infors):
        infor = sum(infors)
        logit = self.layers(infor)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_in_dim', type=int)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_fc_dim', 512)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_in_dim', 128)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class DsodImgNet(Net):
    def __init__(self,
                 in_dim,
                 growth_rate,
                 layer_dims,
                 layer_types,
                 layer_depths,
                 layer_widens,
                 layer_coord_types,
                 layer_norm_types,
                 layer_se_types,
                 ):
        super().__init__()
        self.layer_types = layer_types
        self.layers = nn.ModuleList()
        self.layer_types = layer_types
        for idx in range(len(layer_types)):
            sub_args = argparse.Namespace(in_dim=in_dim, out_dim=layer_dims[idx], block_depth=layer_depths[idx],
                                          growth_rate=growth_rate, widen=layer_widens[idx], se_type=layer_se_types[idx],
                                          norm_type=layer_norm_types[idx], coord_type=layer_coord_types[idx],
                                          )
            layer = Layer.build(sub_args, cls_name=layer_types[idx])
            in_dim = layer_dims[idx]
            self.layers.append(layer)
        self._out_dim = in_dim
        self.reset_parameters()

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, img):
        layer_in_feat = img
        img_feats = list()
        for layer in self.layers:
            layer_in_feat = layer(layer_in_feat)
            img_feats.append(layer_in_feat)
        return img_feats[-1]

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--img_in_dim', type=int)
        group.add_argument('--img_growth_rate', type=int)
        group.add_argument('--img_layer_widens', nargs='*', type=int)
        group.add_argument('--img_layer_depths', nargs='*', type=int)
        group.add_argument('--img_layer_coord_types', nargs='*', type=str)
        group.add_argument('--img_layer_dims', nargs='*', type=int)
        group.add_argument('--img_layer_types', nargs='*', type=str)
        group.add_argument('--img_layer_norm_types', nargs='*', type=str)
        group.add_argument('--img_layer_se_types', nargs='*', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'img_in_dim', 3)
        try_set_attr(params, 'img_growth_rate', 32)
        try_set_attr(params, 'img_layer_types', ('img_init_layer', 'dsod_layer', 'dsod_layer', 'dsod_layer'))
        try_set_attr(params, 'img_layer_dims', (64, 128, 128, 128))
        if hasattr(params, 'img_layer_types'):
            layer_num = len(params.img_layer_types)
            try_set_attr(params, 'img_layer_widens', (1,) * layer_num)
            try_set_attr(params, 'img_layer_coord_types', ('default',) * layer_num)
            try_set_attr(params, 'img_layer_depths', (4,) * layer_num)
            try_set_attr(params, 'img_layer_norm_types', ('batch',) * 4)
            try_set_attr(params, 'img_layer_se_types', ('None',) * 4)
        return params

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='img')


class FilmFusionNet(Net):
    def __init__(self,
                 in_dim,
                 layer_types,
                 layer_dims,
                 layer_norm_types,
                 layer_act_types,
                 layer_fusion_norm_types,
                 layer_coord_types,
                 ):
        super().__init__()
        self.layer_dims = layer_dims
        self.layers = nn.ModuleList()
        for idx in range(len(layer_types)):
            sub_args = argparse.Namespace(in_dim=in_dim, norm_type=layer_norm_types[idx], orders=None,
                                          fusion_norm_type=layer_fusion_norm_types[idx],
                                          act_type=layer_act_types[idx], coord_type=layer_coord_types[idx]
                                          )
            self.layers.append(Layer.build(sub_args, cls_name=layer_types[idx]))
            in_dim = layer_dims[idx]
        self._out_dim = in_dim
        self.reset_parameters()

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, img_feat, q_info):
        gammas, betas = map(lambda x: x.squeeze(), q_info.chunk(2, dim=1))
        gammas, betas = gammas.chunk(len(self.layers), dim=1), betas.chunk(len(self.layers), dim=1)
        in_feat = img_feat
        for layer_idx, layer in enumerate(self.layers):
            # info_slice = slice(*(partial_sum(self.layer_dims, layer_idx + i) for i in range(2)))
            in_feat = layer(in_feat, gammas[layer_idx], betas[layer_idx])
        return in_feat

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--fusion_in_dim', type=int)
        group.add_argument('--fusion_layer_types', nargs='*', type=str)
        group.add_argument('--fusion_layer_dims', nargs='*', type=int)
        group.add_argument('--fusion_layer_norm_types', nargs='*', type=str)
        group.add_argument('--fusion_layer_act_types', nargs='*', type=str)
        group.add_argument('--fusion_layer_fusion_norm_types', nargs='*', type=str)
        group.add_argument('--fusion_layer_coord_types', nargs='*', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'fusion_in_dim', 128)
        try_set_attr(params, 'fusion_layer_types', ('film_fusion_layer',) * 4)
        if hasattr(params, 'fusion_layer_types'):
            layer_num = len(params.fusion_layer_types)
            try_set_attr(params, 'fusion_layer_dims', (params.fusion_in_dim,) * layer_num)
            # try_set_attr(args, 'fusion_layer_dims', (256, 384, 512, 640))
            try_set_attr(params, 'fusion_layer_norm_types', ('batch',) * layer_num)
            try_set_attr(params, 'fusion_layer_act_types', ('relu',) * layer_num)
            try_set_attr(params, 'fusion_layer_fusion_norm_types', ('batch',) * layer_num)
            try_set_attr(params, 'fusion_layer_coord_types', ('default',) * layer_num)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='fusion')


class FilmClassifierNet(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, feat_h, norm_type, coord_type):
        super().__init__()
        if coord_type != 'None':
            proj_l = Conv2D(in_dim, proj_dim, 1, norm_type=norm_type, orders=('coord', 'conv', 'norm', 'act'),
                            coord_type=coord_type)
        else:
            proj_l = Null()
            proj_dim = in_dim
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim if proj_dim > 0 else in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, x):
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_coord_type', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_coord_type', 'default')
        try_set_attr(params, 'classifier_in_dim', 128)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class FilmClassifierNetA(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, feat_h, proj_norm_type, norm_type, coord_type):
        super().__init__()
        if coord_type != 'None':
            proj_l = Conv2D(in_dim, proj_dim, 1, norm_type=proj_norm_type, orders=('coord', 'conv', 'norm', 'act'),
                            coord_type=coord_type)
        else:
            proj_l = Null()
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim if proj_dim > 0 else in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, x):
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)
        group.add_argument('--classifier_proj_norm_type', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_in_dim', 128)
        try_set_attr(params, 'classifier_proj_norm_type', 'instance')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class FilmClassifierNetB(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, feat_h, norm_type, act_type, coord_type):
        super().__init__()
        if coord_type != 'None':
            proj_l = Conv2D(in_dim, proj_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                            coord_type=coord_type)
        else:
            proj_l = Null()
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim if proj_dim > 0 else in_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, x):
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_coord_type', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_coord_type', 'default')
        try_set_attr(params, 'classifier_in_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')









class FilmClassifierNetD(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, feat_h, norm_type, act_type, coord_type):
        super().__init__()
        if coord_type != 'None':
            proj_l = Conv2D(in_dim, proj_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                            coord_type=coord_type)
        else:
            proj_l = Null()
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim if proj_dim > 0 else in_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, x):
        x = x[:, -128:, :, :]
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_coord_type', type=str)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_coord_type', 'default')
        try_set_attr(params, 'classifier_in_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')







class FilmClassifierNetC(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, feat_h, norm_type, add_coord):
        super().__init__()
        if add_coord:
            proj_l = Conv2D(in_dim, proj_dim, 1, norm_type=norm_type, orders=('coord', 'conv', 'act'))
        else:
            proj_l = Conv2D(in_dim, proj_dim, 1, norm_type=norm_type, orders=('conv', 'act'))
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim if proj_dim > 0 else in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, x):
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_in_dim', 128)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class SoftAttnClassifierNet(Net):
    def __init__(self, answer_num, in_dim, proj_dim, fc_dim, norm_type,):
        super().__init__()
        self.attn_l = SoftAttnLayer(in_dim, attn_max_num=0)
        self.layers = nn.Sequential(
            # PtLinear(in_dim, proj_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(in_dim, fc_dim, norm_type=norm_type, orders=('linear', 'norm', 'act')),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        attn_feat = self.attn_l(img_feat, q_query)
        logit = self.layers(attn_feat)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_in_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_in_dim', 128)

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class ConcatClassifierNet(Net):
    def __init__(self, answer_num, img_dim, query_dim, proj_dim, fc_dim, feat_h, norm_type, act_type, add_coord):
        super().__init__()
        if add_coord:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('coord', 'conv', 'norm', 'act'), act_type=act_type,
                            norm_type=norm_type)
        else:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('conv', 'norm', 'act'), act_type=act_type,
                            norm_type=norm_type)
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        x = torch.cat((img_feat, q_query.view(*q_query.shape, 1, 1).expand_as(img_feat)), dim=1)
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_img_dim', type=int)
        group.add_argument('--classifier_query_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_img_dim', 128)
        try_set_attr(params, 'classifier_query_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class ConcatClassifierNetA(Net):
    def __init__(self, answer_num, img_dim, query_dim, proj_dim, fc_dim, feat_h, norm_type, act_type, add_coord):
        super().__init__()
        if add_coord:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('coord', 'conv', 'norm', 'act'), act_type=act_type,
                            norm_type=norm_type)
        else:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('conv', 'act'), act_type=act_type,
                            norm_type=norm_type)
        self.layers = nn.Sequential(
            proj_l,
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
            Linear(proj_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        x = torch.cat((img_feat, q_query.view(*q_query.shape, 1, 1).expand_as(img_feat)), dim=1)
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_img_dim', type=int)
        group.add_argument('--classifier_query_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 29)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_add_coord', False)
        try_set_attr(params, 'classifier_img_dim', 128)
        try_set_attr(params, 'classifier_query_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class HardAttnClassifierNet(Net):
    def __init__(self, answer_num, img_dim, query_dim, proj_dim, fc_dim, feat_h, norm_type, act_type, add_coord):
        super().__init__()
        from .layers.film_layers import HardAttnSumLayer
        if add_coord:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('coord', 'conv', 'norm', 'act'), act_type=act_type,
                            norm_type=norm_type)
        else:
            proj_l = Conv2D(img_dim + query_dim, proj_dim, 1, orders=('conv', 'norm', 'act'), act_type=act_type,
                            norm_type=norm_type)
        self.layers = nn.Sequential(
            proj_l,
            HardAttnSumLayer(),
            Linear(proj_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        x = torch.cat((img_feat, q_query.view(*q_query.shape, 1, 1).expand_as(img_feat)), dim=1)
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_feat_h', type=int)
        group.add_argument('--classifier_img_dim', type=int)
        group.add_argument('--classifier_query_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_feat_h', 14)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_img_dim', 128)
        try_set_attr(params, 'classifier_query_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class HardAttnRnnClassifierNet(Net):
    def __init__(self, answer_num, img_dim, proj_dim, fc_dim, norm_type, act_type, attn_num):
        super().__init__()
        from .layers.film_layers import HardAttnRnnLayer
        self.layers = nn.Sequential(
            HardAttnRnnLayer(img_dim, proj_dim, attn_num),
            Linear(proj_dim, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        # x = torch.cat((img_feat, q_query.view(*q_query.shape, 1, 1).expand_as(img_feat)), dim=1)
        x = img_feat
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_attn_num', type=int)
        group.add_argument('--classifier_img_dim', type=int)
        group.add_argument('--classifier_query_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 512)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_attn_num', 40)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_img_dim', 128)
        try_set_attr(params, 'classifier_query_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')


class HardAttnCatClassifierNet(Net):
    def __init__(self, answer_num, img_dim, query_dim, proj_dim, fc_dim, norm_type, act_type, attn_max):
        super().__init__()
        from .layers.film_layers import HardAttnConcatLayer
        self.layers = nn.Sequential(
            HardAttnConcatLayer(img_dim, proj_dim, attn_max),
            Linear(proj_dim * attn_max, fc_dim, norm_type=norm_type,
                   orders=('linear', 'norm', 'act'), act_type=act_type
                   ),
            Linear(fc_dim, answer_num, orders=('linear',))
        )
        self.reset_parameters()

    def forward(self, img_feat, q_query):
        # x = torch.cat((img_feat, q_query.view(*q_query.shape, 1, 1).expand_as(img_feat)), dim=1)
        x = img_feat
        logit = self.layers(x)
        return logit

    @classmethod
    def add_args(cls, group: argparse.ArgumentParser, params=None, cls_name=None, sub_cls=None):
        group.add_argument('--classifier_answer_num', type=int)
        group.add_argument('--classifier_proj_dim', type=int)
        group.add_argument('--classifier_fc_dim', type=int)
        group.add_argument('--classifier_norm_type', type=str)
        group.add_argument('--classifier_act_type', type=str)
        group.add_argument('--classifier_attn_max', type=int)
        group.add_argument('--classifier_img_dim', type=int)
        group.add_argument('--classifier_query_dim', type=int)
        group.add_argument('--classifier_add_coord', type=str_bool)

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, 'classifier_answer_num', 28)
        try_set_attr(params, 'classifier_proj_dim', 30)
        try_set_attr(params, 'classifier_fc_dim', 1024)
        try_set_attr(params, 'classifier_norm_type', 'batch')
        try_set_attr(params, 'classifier_attn_max', 50)
        try_set_attr(params, 'classifier_add_coord', True)
        try_set_attr(params, 'classifier_img_dim', 128)
        try_set_attr(params, 'classifier_query_dim', 128)
        try_set_attr(params, 'classifier_act_type', 'relu')

    @classmethod
    def build(cls, params, net_cls_name=None, net_cls=None):
        return cls.default_build(cls, params, prefix='classifier')






















































