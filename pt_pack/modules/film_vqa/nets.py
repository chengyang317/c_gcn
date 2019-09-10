# coding=utf-8
from pt_pack.modules.nets import LayerNet, Net
from pt_pack.modules.layers import Linear
from pt_pack.utils import try_set_attr, load_func_kwargs
import torch.nn as nn
from .layers import CoordLayer


__all__ = ['FilmQNet', 'FilmImgNet', 'FilmFusionNet', 'FilmClsNet']


class FilmQNet(Net):
    prefix = 'film_q_net'

    def __init__(self,
                 vocab_num: int,
                 embed_dim: int,
                 out_dim: int,
                 is_bi: bool=False):
        super().__init__()

        self.embed_l = nn.Embedding(vocab_num, embed_dim)
        self.encoder_l = nn.GRU(embed_dim, out_dim if not is_bi else out_dim//2, batch_first=True, bidirectional=is_bi)
        self.reset_parameters()

    def forward(self, q_labels, q_lens):
        q_embed = self.embed_l(q_labels)
        self.encoder_l.flatten_parameters()
        seq_out, _ = self.encoder_l(q_embed)
        packed = nn.utils.rnn.pack_padded_sequence(q_embed, q_lens.squeeze().tolist(), batch_first=True)
        _, hid = self.encoder_l(packed)
        return hid.squeeze()

    @classmethod
    def init_args(cls, params, cls_name=None, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_vocab_num', 82)
        try_set_attr(params, f'{cls.prefix_name()}_embed_dim', 100)
        try_set_attr(params, f'{cls.prefix_name()}_out_dim', 2048)
        try_set_attr(params, f'{cls.prefix_name()}_is_bi', False)


class FilmImgNet(LayerNet):
    prefix = 'film_img_net'

    def __init__(self,
                 layers,
                 ):
        super().__init__(layers)
        self.reset_parameters()

    def forward(self, img_feats):
        for layer in self.layers:
            layer_kwargs = load_func_kwargs({'img_feats': img_feats}, layer.forward)
            img_feats = layer(**layer_kwargs)
        return img_feats

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_layers', ('img_init_layer_a', 'dsod_layer', 'dsod_layer'))
        try_set_attr(params, f'{cls.prefix_name()}_layer_in_dims', (3, 64, 128))
        try_set_attr(params, f'{cls.prefix_name()}_layer_out_dims', (64, 128, 128))


class FilmFusionNet(LayerNet):
    prefix = 'film_fusion_net'

    def __init__(self, layers):
        super().__init__(layers)
        self.reset_parameters()

    def forward(self, img_feats, q_feats):
        for layer in self.layers:
            layer_kwargs = load_func_kwargs({'img_feats': img_feats, 'q_feats': q_feats}, layer.forward)
            img_feats = layer(**layer_kwargs)
        return img_feats

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_layers',
                     ('dense_film_layer', 'dense_film_layer', 'dense_film_layer', 'dense_film_layer'))
        try_set_attr(params, f'{cls.prefix_name()}_layer_img_dims', (128, 256, 384, 512))


class FilmClsNet(Net):
    prefix = 'film_cls_net'

    def __init__(self,
                 img_dim: int=640,
                 q_dim: int=2048,
                 hid_dim: int=1024,
                 answer_num: int=28,
                 norm_type: str='batch',
                 coord_type: str='default'
                 ):
        super().__init__()
        self.img_dim = img_dim
        self.img_proj = CoordLayer(img_dim, hid_dim//2, coord_type)
        self.linear_l = nn.Sequential(
            Linear(img_dim if coord_type is None else hid_dim//2, hid_dim, norm_type=norm_type),
            nn.Linear(hid_dim, answer_num)
        )
        self.reset_parameters()

    def forward(self, img_feats, q_feats):
        img_feats = img_feats[:, -self.img_dim:, :, :]
        img_feats = self.img_proj(img_feats)
        b_size, o_c, _, _ = img_feats.shape
        img_feats = img_feats.view(b_size, o_c, -1).max(-1)[0]
        logit = self.linear_l(img_feats)
        return logit


class FilmClsNetA(Net):
    prefix = 'film_cls_net'

    def __init__(self,
                 img_dim: int=640,
                 q_dim: int=4096,
                 hid_dim: int=1024,
                 answer_num: int=28,
                 norm_type: str='batch',
                 coord_type: str='default'
                 ):
        super().__init__()
        self.img_proj = CoordLayer(img_dim, hid_dim//2, coord_type)
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(img_dim if coord_type is None else hid_dim//2, hid_dim)),
            nn.utils.weight_norm(nn.Linear(hid_dim, answer_num))
        )
        self.reset_parameters()

    def forward(self, img_feats, q_feats):
        img_feats = self.img_proj(img_feats)
        b_size, o_c, _, _ = img_feats.shape
        img_feats = img_feats.view(b_size, o_c, -1).max(-1)[0]
        logit = self.linear_l(img_feats)
        return logit


class FilmClsNetB(Net):
    prefix = 'film_cls_net'

    def __init__(self,
                 img_dim: int=640,
                 q_dim: int=4096,
                 hid_dim: int=1024,
                 answer_num: int=28,
                 norm_type: str='batch',
                 coord_type: str='default'
                 ):
        super().__init__()
        # self.img_proj = CoordLayer(img_dim, hid_dim//2, coord_type)
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(img_dim, hid_dim)),
            nn.utils.weight_norm(nn.Linear(hid_dim, answer_num))
        )
        self.reset_parameters()

    def forward(self, img_feats, q_feats):
        # img_feats = self.img_proj(img_feats)
        b_size, o_c, _, _ = img_feats.shape
        img_feats = img_feats.view(b_size, o_c, -1).max(-1)[0]
        logit = self.linear_l(img_feats)
        return logit


class FilmClsNetC(Net):
    prefix = 'film_cls_net'

    def __init__(self,
                 img_dim: int=480,
                 q_dim: int=1024,
                 hid_dim: int=1024,
                 answer_num: int=28,
                 norm_type: str='batch',
                 coord_type: str='default'
                 ):
        super().__init__()
        # self.img_proj = CoordLayer(img_dim, hid_dim//2, coord_type)
        self.q_linear = nn.utils.weight_norm(nn.Linear(q_dim, img_dim))
        self.relu_l = nn.ReLU()
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(img_dim, hid_dim)),
            nn.utils.weight_norm(nn.Linear(hid_dim, answer_num))
        )
        self.reset_parameters()

    def forward(self, img_feats, q_feats):
        # img_feats = self.img_proj(img_feats)
        b_size, o_c, _, _ = img_feats.shape
        img_feats = img_feats.view(b_size, o_c, -1).max(-1)[0]
        img_feats = img_feats * self.relu_l(self.q_linear(q_feats))
        logit = self.linear_l(img_feats)
        return logit

