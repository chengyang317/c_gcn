# coding=utf-8
from pt_pack.modules.layers import Layer, Conv2D, Null
import torch.nn as nn
import torch

__all__ = ['ImgInitLayer', 'DsodLayer', 'FilmLayer', 'ImgInitLayerA']


class DenseLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 growth_rate,
                 widen,
                 dropout,
                 norm_type,
                 orders,
                 ):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2D(in_dim, int(widen * growth_rate), 1, orders=orders, norm_type=norm_type),
            Conv2D(int(widen * growth_rate), growth_rate, 3, padding=1, norm_type=norm_type, orders=orders),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        left_x = x
        right_x = self.layers(x)
        return torch.cat((left_x, right_x), dim=1)


class DenseBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 depth,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act')
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.block_depth = depth
        self.growth_rate = growth_rate
        self.layers = nn.Sequential(
            *[DenseLayer(in_dim + idx * growth_rate, growth_rate, widen, dropout, norm_type, orders) for idx in range(depth)]
        )

    @property
    def out_dim(self):
        return self.in_dim + self.growth_rate * self.block_depth

    def forward(self, x):
        return self.layers(x)


class CoordLayer(nn.Module):
    def __init__(self, in_dim, out_dim, coord_type):
        super().__init__()
        if coord_type is not None:
            self.layer = Conv2D(in_dim, out_dim, orders=('coord', 'conv', 'act'), coord_type=coord_type)
        else:
            self.layer = Null()

    def forward(self, x):
        return self.layer(x)


class ImgInitLayer(Layer):
    prefix = 'img_init_layer'

    def __init__(self,
                 in_dim: int=3,
                 out_dim: int=64,
                 widen: int=1,
                 norm_type: str='batch',
                 coord_type: str=None,
                 se_type: str=None,
                 ):
        super().__init__()
        se_orders = ('conv', 'norm', 'se', 'act') if se_type is not None else ('conv', 'norm', 'act')
        self.layers = nn.Sequential(
            Conv2D(in_dim, out_dim * widen, 3, stride=2, padding=1, norm_type=norm_type),
            Conv2D(out_dim * widen, out_dim, 3, padding=1, norm_type=norm_type),
            CoordLayer(out_dim, out_dim, coord_type=coord_type),
            Conv2D(out_dim, out_dim, 3, padding=1, norm_type=norm_type, orders=se_orders, se_type=se_type),
        )

    def forward(self, img_feats):
        return self.layers(img_feats)


class ImgInitLayerA(Layer):
    prefix = 'img_init_layer'

    def __init__(self,
                 in_dim: int=3,
                 out_dim: int=64,
                 widen: int=1,
                 norm_type: str='batch',
                 coord_type: str=None,
                 se_type: str=None,
                 ):
        super().__init__()
        se_orders = ('conv', 'norm', 'se', 'act') if se_type is not None else ('conv', 'norm', 'act')
        self.layers = nn.Sequential(
            Conv2D(in_dim, out_dim//2, 3, stride=2, padding=1, norm_type=norm_type),
            Conv2D(out_dim//2, out_dim, 3, padding=1, norm_type=norm_type, orders=se_orders, se_type=se_type),
        )

    def forward(self, img_feats):
        return self.layers(img_feats)


class ImgInitLayerB(Layer):
    prefix = 'img_init_layer'

    def __init__(self,
                 in_dim: int=3,
                 out_dim: int=64,
                 widen: int=1,
                 norm_type: str='batch',
                 coord_type: str=None,
                 se_type: str=None,
                 ):
        super().__init__()
        se_orders = ('conv', 'norm', 'se', 'act') if se_type is not None else ('conv', 'norm', 'act')
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            Conv2D(out_dim//2, out_dim, 3, padding=1, norm_type=norm_type, orders=se_orders, se_type=se_type),
        )

    def forward(self, img_feats):
        return self.layers(img_feats)


class DsodLayer(Layer):
    prefix = 'dsod_layer'

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 block_depth: int=4,
                 growth_rate: int=48,
                 widen: int=1,
                 dropout: float=0.,
                 norm_type: str='batch',
                 coord_type: str=None,
                 se_type: str=None,
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        se_orders = ('conv', 'norm', 'se', 'act') if se_type is not None else ('conv', 'norm', 'act')
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 3, 2, 1, norm_type=norm_type),
            CoordLayer(in_dim//2, in_dim//2, coord_type=coord_type),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type),
            Conv2D(block_out_dim, out_dim // 2, orders=se_orders, norm_type=norm_type, se_type=se_type)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim // 2, orders=se_orders, norm_type=norm_type, se_type=se_type)
        )
        self.out_dim = out_dim

    def forward(self, img_feats):
        return torch.cat((self.left_l(img_feats), self.right_l(img_feats)), dim=1)


class FilmLayer(Layer):
    prefix = 'film_layer'

    def __init__(self,
                 img_dim: int,
                 q_dim: int,
                 out_dim: int,
                 norm_type: str='instance',
                 coord_type: str='default'
                 ):
        super().__init__()
        self.img_proj = CoordLayer(img_dim, img_dim, coord_type)
        self.q_proj = nn.Linear(q_dim, out_dim*2)
        self.cond_conv = Conv2D(img_dim, out_dim, 3, padding=1, norm_type=norm_type, norm_affine=False,
                                orders=('conv', 'norm', 'cond', 'act'))

    def forward(self, img_feats, q_feats):
        img_feats = self.img_proj(img_feats)
        gammas, betas = self.q_proj(q_feats).chunk(2, dim=-1)
        gammas += 1.0
        residual = self.cond_conv(img_feats, gammas, betas)
        out = residual + img_feats
        return out


class DenseFilmLayer(Layer):
    prefix = 'dense_film_layer'

    def __init__(self,
                 img_dim: int,
                 q_dim: int=2048,
                 hid_dim: int=128,
                 norm_type: str='instance',
                 coord_type: str='default'
                 ):
        super().__init__()
        self.img_proj = CoordLayer(img_dim, hid_dim, coord_type) if coord_type is not None else Conv2D(img_dim, hid_dim)
        self.q_proj = nn.Linear(q_dim, hid_dim * 2)
        self.cond_l = Conv2D(img_dim if coord_type is None else hid_dim, hid_dim, 3, padding=1, norm_type=norm_type,
                             norm_affine=False, orders=('conv', 'norm', 'cond', 'act'))

    def forward(self, img_feats, q_feats):
        img_proj_feats = self.img_proj(img_feats)
        gammas, betas = self.q_proj(q_feats).chunk(2, dim=-1)
        gammas += 1.0
        residual = self.cond_l(img_proj_feats, gammas, betas)
        out = torch.cat((img_feats, residual), dim=1)
        return out




