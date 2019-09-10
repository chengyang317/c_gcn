# coding=utf-8
from .base_layers import Layer, Conv2D, Null, Flatten, Se
import torch.nn as nn
import torch
from ..fair_seq import MultiheadAttention
import argparse


class DenseLayer(Layer):
    def __init__(self,
                 in_dim,
                 growth_rate,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act'),
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


class DenseBlock(Layer):
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


class DsodLayer(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth=4,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act'),
                 coord_type='default',
                 se_type='None',
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        se_orders = ('conv', 'norm', 'se', 'act')
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 3, 2, 1, orders=orders, norm_type=norm_type),
            Conv2D(in_dim // 2, in_dim // 2, orders=('coord', 'conv', 'act'),
                   coord_type=coord_type) if coord_type != 'None' else Null(),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type, orders),
            Conv2D(block_out_dim, out_dim // 2, orders=orders if se_type == 'None' else se_orders,
                   norm_type=norm_type, se_type=se_type)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim // 2, orders=orders if se_type == 'None' else se_orders,
                   norm_type=norm_type, se_type=se_type)
        )
        self.out_dim = out_dim

    def forward(self, x):
        return torch.cat((self.left_l(x), self.right_l(x)), dim=1)


class DsodLayerB(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth=4,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 add_coord=False,
                 orders=('norm', 'act', 'conv'),
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 1, bias=False, orders=orders),
            Conv2D(in_dim // 2, in_dim // 2, 3, 2, 1, bias=False, orders=orders),
            Null() if not add_coord else Conv2D(in_dim // 2, in_dim // 2, 1, orders=('coord', 'conv')),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type, orders),
            Conv2D(block_out_dim, out_dim // 2, 1, orders=orders)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim // 2, 1, orders=orders)
        )
        self.out_dim = out_dim

    def forward(self, x):
        return torch.cat((self.left_l(x), self.right_l(x)), dim=1)


class DsodLayerC(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth=4,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act'),
                 coord_type='default',
                 se_type='None',
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        se_orders = ('conv', 'norm', 'se', 'act')
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 3, 2, 1, orders=orders, norm_type=norm_type),
            Conv2D(in_dim // 2, in_dim // 2, orders=('coord', 'conv', 'act'),
                   coord_type=coord_type) if coord_type != 'None' else Null(),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type, orders),
            Conv2D(block_out_dim, out_dim, orders=orders if se_type == 'None' else se_orders,
                   norm_type=norm_type, se_type=se_type)
        )
        # self.left_l = nn.Sequential(
        #     nn.MaxPool2d(2, 2, ceil_mode=True),
        #     PtConv2d(in_dim, out_dim // 2, orders=orders if se_type == 'None' else se_orders,
        #              norm_type=norm_type, se_type=se_type)
        # )
        self.out_dim = out_dim

    def forward(self, x):
        # return torch.cat((self.left_l(x), self.right_l(x)), dim=1)
        return self.right_l(x)


class DsodLayerD(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth=4,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act'),
                 coord_type='default',
                 se_type='None',
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        se_orders = ('conv', 'norm', 'se', 'act')
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 3, 2, 1, orders=orders, norm_type=norm_type),
            Conv2D(in_dim // 2, in_dim // 2, orders=('coord', 'conv', 'act'),
                   coord_type=coord_type) if coord_type != 'None' else Null(),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type, orders),
            Conv2D(block_out_dim, out_dim, orders=orders if se_type == 'None' else se_orders,
                   norm_type=norm_type, se_type=se_type)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim, orders=orders if se_type == 'None' else se_orders,
                   norm_type=norm_type, se_type=se_type)
        )
        self.out_dim = out_dim

    def forward(self, x):
        # return torch.cat((self.left_l(x), self.right_l(x)), dim=1)
        return self.right_l(x) + self.left_l(x)


class DsodFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_h
                 ):
        super().__init__()
        self.img_layers = DsodLayer(in_dim, out_dim, add_coord=True)
        self.cond_layer = Conv2D(out_dim, out_dim, 3, padding=1, norm_type='instance', norm_affine=False,
                                 orders=('conv', 'norm', 'cond', 'act'))
        self.conv_layers = nn.Sequential(
            Conv2D(out_dim, out_dim // 4, 1, orders=('conv', 'act')),
            Conv2D(out_dim // 4, out_dim, 1, orders=('conv', 'act'))
        )
        self.max_l = nn.Sequential(
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
        )

    def forward(self, img, gamma, beta):
        img = self.img_layers(img)
        out = self.cond_layer(img, gamma, beta)
        out = self.conv_layers(out)
        out = out + img
        return img, self.max_l(out)


class DsodFusionLayerB(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_h,
                 block_num=4,
                 ):
        super().__init__()
        self.img_layers = DsodLayer(in_dim, out_dim, add_coord=True)
        self.cond_n = nn.ModuleList()
        for _ in range(block_num):
            self.cond_n.append(
                Conv2D(out_dim, out_dim, 3, padding=1, norm_type='instance', norm_affine=False,
                       orders=('conv', 'norm', 'cond', 'act'))
            )

        self.max_l = nn.Sequential(
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
        )

    def forward(self, img, gamma, beta):
        img = self.img_layers(img)
        out = self.cond_layer(img, gamma, beta)
        out = self.conv_layers(out)
        out = out + img
        return img, self.max_l(out)


class DsodFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_h
                 ):
        super().__init__()
        self.img_layers = DsodLayer(in_dim, out_dim, add_coord=True)
        self.cond_layer = FilmFusionLayer(out_dim)
        self.max_l = nn.Sequential(
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
        )

    def forward(self, img, gamma, beta):
        img = self.img_layers(img)
        out = self.cond_layer(img, gamma, beta)
        return img, self.max_l(out)


class PyramidFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 feat_h,
                 block_num=2,
                 add_coord=True,
                 layer_type='film_fusion_layer'
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for _ in range(block_num):
            sub_args = argparse.Namespace(in_dim=in_dim, add_coord=add_coord)
            self.layers.append(Layer.build(sub_args, layer_type))
        self.max_l = nn.Sequential(
            nn.MaxPool2d(feat_h, feat_h, 0),
            Flatten(),
        )

    def forward(self, img_feat, gammas, betas):
        gammas, betas = gammas.chunk(self.block_num, dim=1), betas.chunk(self.block_num, dim=1)
        for idx, layer in enumerate(self.layers):
            img_feat = layer(img_feat, gammas[idx], betas[idx])
        return self.max_l(img_feat)


class DsodLayerA(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 add_coord=False,
                 orders=('conv', 'norm', 'act'),
                 ):
        super().__init__()
        block_out_dim = in_dim//2 + block_depth * growth_rate
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim // 2, 3, 2, 1, bias=False, orders=orders),
            Null() if not add_coord else Conv2D(in_dim // 2, in_dim // 2, 1, orders=('coord', 'conv', 'act')),
            DenseBlock(in_dim // 2, block_depth, growth_rate, widen, dropout, norm_type, orders),
            Conv2D(block_out_dim, out_dim, 1, orders=orders)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim, 1, orders=orders)
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.left_l(x) + self.right_l(x)


class ResDenseLayer(Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 block_depth,
                 growth_rate=48,
                 widen=1,
                 dropout=0.,
                 norm_type='batch',
                 add_coord=True,
                 orders=('conv', 'norm', 'act'),
                 ):
        super().__init__()
        block_out_dim = in_dim + block_depth * growth_rate
        block_orders = ['coord'] + list(orders) if add_coord else orders
        self.right_l = nn.Sequential(
            Conv2D(in_dim, in_dim, 3, 2, 1, bias=False, orders=orders),
            DenseBlock(in_dim, block_depth, growth_rate, widen, dropout, norm_type, block_orders),
            Conv2D(block_out_dim, out_dim, 1, orders=orders)
        )
        self.left_l = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            Conv2D(in_dim, out_dim, 1, orders=orders)
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.left_l(x) + self.right_l(x)


class ImgInitLayer(Layer):
    def __init__(self,
                 in_dim=3,
                 out_dim=64,
                 widen=1,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act'),
                 coord_type='default',
                 se_type='None',
                 ):
        super().__init__()
        se_orders = ('conv', 'norm', 'se', 'act')
        self.stem_l = nn.Sequential(
            nn.Sequential(
                Conv2D(in_dim, out_dim * widen, 3, stride=2, padding=1, norm_type=norm_type, orders=orders),
                Conv2D(out_dim * widen, out_dim, 3, padding=1, norm_type=norm_type, orders=orders),
                Conv2D(out_dim, out_dim, 1, orders=('coord', 'conv', 'act'),
                       coord_type=coord_type) if coord_type != 'None' else Null(),
                Conv2D(out_dim, out_dim, 3, padding=1, norm_type=norm_type,
                       orders=orders if se_type == 'None' else se_orders, se_type=se_type),
            )
        )

    def forward(self, x):
        return self.stem_l(x)


class ImgInitLayerA(Layer):
    def __init__(self,
                 in_dim=3,
                 out_dim=64,
                 widen=1,
                 add_coord=True,
                 norm_type='batch',
                 orders=('conv', 'norm', 'act')
                 ):
        super().__init__()
        self.conv_1 = Conv2D(in_dim, out_dim // 2, 1, norm_type=norm_type, orders=orders)
        self.conv_3 = Conv2D(in_dim, out_dim // 2, 3, padding=1, norm_type=norm_type, orders=orders)
        self.conv_5 = Conv2D(in_dim, out_dim // 2, 5, padding=2, norm_type=norm_type, orders=orders)
        self.conv_7 = Conv2D(in_dim, out_dim // 2, 7, padding=3, norm_type=norm_type, orders=orders)
        self.weight = nn.Parameter(torch.ones(4))
        self.pool_l = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        fine = self.weight[0] * self.conv_1(x) + self.weight[1] * self.conv_3(x)
        coarse = self.weight[2] * self.conv_5(x) + self.weight[-1] * self.conv_7(x)
        return torch.cat((self.pool_l(fine), self.pool_l(coarse)), dim=1)


class FilmFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                               coord_type=coord_type) if coord_type != 'None' else Null()
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                                orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(img_feat, gammas, betas)
        out = residual + img_feat
        return out


class DenseFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.proj_l = Conv2D(in_dim, in_dim // 2, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                             coord_type=coord_type) if add_coord else Null()
        self.cond_l = Conv2D(in_dim // 2, in_dim // 2, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                             orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.proj_l(img_feat)
        residual = self.cond_l(img_feat, gammas, betas)
        out = torch.cat((img_feat, residual), dim=1)
        return out


class DenseFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.proj_l = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                             coord_type=coord_type) if add_coord else Null()
        self.cond_l = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                             orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)
        self.conv_l = Conv2D(in_dim * 2, in_dim)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.proj_l(img_feat)
        residual = self.cond_l(img_feat, gammas, betas)
        out = torch.cat((img_feat, residual), dim=1)
        out = self.conv_l(out)
        return out


class DenseFilmFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 coord_type='default'
                 ):
        super().__init__()
        self.proj_l = Conv2D(in_dim, 128, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                             coord_type=coord_type) if coord_type != 'None' else Null()
        self.cond_l = Conv2D(128, 128, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                             orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat_proj = self.proj_l(img_feat)
        residual = self.cond_l(img_feat_proj, gammas, betas)
        out = img_feat_proj + residual
        out = torch.cat((img_feat, out), dim=1)
        return out


class DenseFilmFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 coord_type='default'
                 ):
        super().__init__()
        if coord_type == 'None':
            self.proj_l = Conv2D(in_dim, 128, act_type=act_type)
        else:
            self.proj_l = Conv2D(in_dim, 128, orders=('coord', 'conv', 'act'), act_type=act_type, coord_type=coord_type)
        # self.proj_l = PtConv2d(in_dim, 128, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
        #                        coord_type=coord_type) if coord_type != 'None' else PtNull()
        self.cond_l = Conv2D(128, 128, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                             orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat_proj = self.proj_l(img_feat)
        residual = self.cond_l(img_feat_proj, gammas, betas)
        out = torch.cat((img_feat, residual), dim=1)
        return out


class DenseFilmFusionLayerB(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 coord_type='default'
                 ):
        super().__init__()
        if coord_type == 'None':
            self.proj_l = Conv2D(in_dim, 128, act_type=act_type)
        else:
            self.proj_l = Conv2D(in_dim, 128, orders=('coord', 'conv', 'norm', 'act'), act_type=act_type,
                                 coord_type=coord_type)
        # self.proj_l = PtConv2d(in_dim, 128, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
        #                        coord_type=coord_type) if coord_type != 'None' else PtNull()
        self.cond_l = Conv2D(128, 128, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                             orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat_proj = self.proj_l(img_feat)
        residual = self.cond_l(img_feat_proj, gammas, betas)
        out = torch.cat((img_feat, residual), dim=1)
        return out


class FilmFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False, orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)
        self.layers = nn.Sequential(
            Conv2D(in_dim, in_dim, kernel_size=1, orders=('conv', 'act'), act_type=act_type),
            Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type,
                   coord_type=coord_type) if add_coord else Null()
        )

    def forward(self, img_feat, gammas, betas):
        residual = self.cond_conv(img_feat, gammas, betas)
        residual = self.layers(residual)
        out = residual + img_feat
        return out



class FilmFusionLayerB(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv'), act_type=act_type, coord_type=coord_type) if add_coord else Null()
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False, orders=('act', 'conv', 'norm', 'cond'), act_type=act_type)
        self.relu = nn.ReLU(True)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(img_feat, gammas, betas)
        out = residual + img_feat
        return self.relu(out)


class FilmFusionLayerC(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type, coord_type=coord_type) if add_coord else Null()
        self.cond_l = Conv2D(in_dim, in_dim, orders=('cond', 'act'), act_type=act_type)
        self.conv_n = nn.Sequential(
            Conv2D(in_dim, in_dim // 4, kernel_size=3, padding=1, orders=('conv', 'act')),
            Conv2D(in_dim // 4, in_dim, kernel_size=3, padding=1, orders=('conv', 'act')),
        )

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_l(img_feat, gammas, betas)
        residual = self.conv_n(residual)
        out = residual + img_feat
        return out


class FilmFusionLayerD(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type, coord_type=coord_type) if add_coord else Null()
        self.cond_l = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False, orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)
        self.conv_n = nn.Sequential(
            Conv2D(in_dim, in_dim // 4, kernel_size=3, padding=1, orders=('conv', 'act')),
            Conv2D(in_dim // 4, in_dim, kernel_size=3, padding=1, orders=('conv', 'act')),
        )

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_l(img_feat, gammas, betas)
        residual = self.conv_n(residual)
        out = residual + img_feat
        return out


class FilmFusionLayerE(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type, coord_type=coord_type) if add_coord else Null()
        self.cond_l = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False, orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)
        self.conv_n = nn.Sequential(
            Conv2D(in_dim, in_dim // 4, kernel_size=1, orders=('conv', 'act')),
            Conv2D(in_dim // 4, in_dim, kernel_size=1, orders=('conv', 'act')),
        )

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_l(img_feat, gammas, betas)
        residual = self.conv_n(residual)
        out = residual + img_feat
        return out


class FilmFusionLayerF(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default',
                 cond_block_num=4,
                 reduction=4,
                 ):
        super().__init__()
        if add_coord:
            self.proj_l = Conv2D(in_dim, in_dim // reduction, 1, orders=('coord', 'conv', 'act'), act_type=act_type, coord_type=coord_type)
        else:
            self.proj_l = Conv2D(in_dim, in_dim // reduction, orders=('conv', 'act'))
        self.cond_n = nn.ModuleList()
        for _ in range(cond_block_num):
            self.cond_n.append(
                Conv2D(in_dim // reduction, in_dim // reduction, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                       orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)
            )
        self.cond_block_num = cond_block_num
        self.dim_per_block = int(in_dim / cond_block_num)
        self.post_proj = Conv2D(in_dim // 4, in_dim, orders=('conv', 'act'))

    def forward(self, img_feat, gammas, betas):
        residual = img_feat = self.proj_l(img_feat)
        for layer_id, layer in enumerate(self.cond_n):
            info_slice = slice(*((layer_id+idx)*self.dim_per_block for idx in range(2)))
            residual = layer(residual, gammas[:, info_slice], betas[:, info_slice])
        out = residual + img_feat
        return self.post_proj(out)


class SeFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv'), act_type=act_type, coord_type=coord_type)
        self.cond_conv = Conv2D(in_dim * 2, in_dim, 3, padding=1, orders=('conv', 'se'))
        self.relu = nn.ReLU(True)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(torch.cat((img_feat, gammas.view(*gammas.shape, 1, 1).expand_as(img_feat)), dim=1))
        out = residual + img_feat
        return self.relu(out)


class SeFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = nn.Sequential(
            Conv2D(in_dim * 2, in_dim, 1, orders=('coord', 'conv', 'act')),
            Conv2D(in_dim, in_dim, 3, padding=1, orders=('conv',))
        )
        self.se_l = Se(in_dim)
        self.relu = nn.ReLU(True)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(torch.cat((img_feat, gammas.view(*gammas.shape, 1, 1).expand_as(img_feat)), dim=1))
        residual = self.se_l(img_feat)
        out = residual + img_feat
        return self.relu(out)


class SeFusionLayerB(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.img_proj = nn.Sequential(
            Conv2D(in_dim * 2, in_dim, 1, orders=('coord', 'conv', 'act')),
            Conv2D(in_dim, in_dim, 3, padding=1, orders=('conv',))
        )
        self.se_l = Se(in_dim)
        self.relu = nn.ReLU(True)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(torch.cat((img_feat, gammas.view(*gammas.shape, 1, 1).expand_as(img_feat)), dim=1))
        residual = self.se_l(img_feat)
        out = residual + img_feat
        return self.relu(out)


class FilmFusionPlusLayer(FilmFusionLayer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__(in_dim, act_type, fusion_norm_type, add_coord, coord_type)
        self.post_layers = nn.Sequential(
            Conv2D(in_dim, in_dim, 3, padding=1),
            Conv2D(in_dim, in_dim, 3, padding=1),
            Conv2D(in_dim, in_dim, 3, padding=1)
        )

    def forward(self, img_feat, gammas, betas):
        img_feat = super().forward(img_feat, gammas, betas)
        img_feat = self.post_layers(img_feat)
        return img_feat


class FilmFusionPlusLayer_1(FilmFusionLayer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__(in_dim, act_type, fusion_norm_type, add_coord, coord_type)
        self.post_layers = nn.Sequential(
            Conv2D(in_dim, in_dim, 1),
            Conv2D(in_dim, in_dim, 1),
            Conv2D(in_dim, in_dim, 1)
        )

    def forward(self, img_feat, gammas, betas):
        img_feat = super().forward(img_feat, gammas, betas)
        img_feat = self.post_layers(img_feat)
        return img_feat


class FilmFusionGroupLayer(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 layer_num=4,
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            layer = FilmFusionLayer(in_dim, act_type, fusion_norm_type, add_coord, coord_type)
            self.layers.append(layer)

    def forward(self, img_feat, gammas, betas):
        for layer in self.layers:
            img_feat = layer(img_feat, gammas, betas)
        return img_feat


class FilmFusionGroupLayer_1(Layer):
    def __init__(self,
                 in_dim,
                 act_type='relu',
                 fusion_norm_type='instance',
                 layer_num=2,
                 add_coord=True,
                 coord_type='default'
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layer_num):
            layer = FilmFusionPlusLayer_1(in_dim, act_type, fusion_norm_type, add_coord, coord_type)
            self.layers.append(layer)

    def forward(self, img_feat, gammas, betas):
        for layer in self.layers:
            img_feat = layer(img_feat, gammas, betas)
        return img_feat


class ResFusionLayer(Layer):
    def __init__(self,
                 in_dim,
                 norm_type='batch',
                 act_type='relu',
                 fusion_norm_type='instance',
                 add_coord=True,
                 ):
        super().__init__()
        self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'act'), act_type=act_type) if add_coord else Null()
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False, orders=('conv', 'norm', 'cond', 'act'), act_type=act_type)

    def forward(self, img_feat, gammas, betas):
        img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(img_feat, gammas, betas)
        out = residual + img_feat
        return out


class ResFusionLayerA(Layer):
    def __init__(self,
                 in_dim,
                 norm_type='batch',
                 fusion_norm_type='instance',
                 add_coord=True,
                 ):
        super().__init__()
        if add_coord:
            self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'norm', 'act'), norm_type=norm_type)
        else:
            self.img_proj = None
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                                orders=('conv', 'norm', 'cond', 'act'))

    def forward(self, img_feat, gammas, betas):
        if self.img_proj is not None:
            img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(img_feat, gammas, betas)
        out = residual + img_feat
        return out


class ResFusionLayerB(Layer):
    def __init__(self,
                 in_dim,
                 norm_type='batch',
                 fusion_norm_type='instance',
                 add_coord=True,
                 ):
        super().__init__()
        if add_coord:
            self.img_proj = Conv2D(in_dim, in_dim, 1, orders=('coord', 'conv', 'norm', 'act'), norm_type=norm_type)
        else:
            self.img_proj = None
        self.cond_conv = Conv2D(in_dim, in_dim, 3, padding=1, norm_type=fusion_norm_type, norm_affine=False,
                                orders=('conv', 'norm', 'cond', 'act'))

    def forward(self, img_feat, gammas, betas):
        if self.img_proj is not None:
            img_feat = self.img_proj(img_feat)
        residual = self.cond_conv(img_feat, gammas, betas)
        out = residual + img_feat
        return out


class HardAttnLayer(Layer):
    def __init__(self, attn_max=50):
        super().__init__()
        self.attn_max = attn_max

    def forward(self, x):
        b, c, h, w = x.shape
        with torch.no_grad():
            x_norm = x.view(b, c, -1).norm(dim=1)
            x_sm = x_norm.softmax(dim=-1)
        if self.attn_max is not None:
            _, x_top_inds = x_sm.topk(self.attn_max, dim=-1)
            return x_top_inds
        else:
            x_mask = x_sm.ge(1. / h / w)
            return x_mask


class HardAttnSumLayer(HardAttnLayer):
    def __init__(self, attn_max=None):
        super().__init__(attn_max)

    def forward(self, x, need_mask=False):
        x_mask = super().forward(x)
        x = x.view(*x.shape[:2], -1)  # b, c, n
        x = x * x_mask.unsqueeze(1).expand_as(x).float()
        x = x.sum(dim=-1)
        return x


class HardAttnConcatLayer(HardAttnLayer):
    def __init__(self, in_dim, proj_dim, attn_max=50):
        super().__init__(attn_max)
        self.proj_dim = proj_dim
        self.proj_l = Conv2D(in_dim, proj_dim, 1, orders=('coord', 'conv', 'act'))

    @property
    def out_dim(self):
        return self.attn_max * self.proj_dim

    def forward(self, x):
        x_top_inds = super().forward(x)  # b, n
        x = self.proj_l(x)
        x = x.view(*x.shape[:2], -1)
        x = x.gather(-1, x_top_inds.unsqueeze(1).expand(*x.shape[:2], self.attn_max))  # b, c, max_num
        return x.view(x.shape[0], -1)


class SoftAttnLayer(Layer):
    def __init__(self, in_dim, head_num=4, attn_max_num=-1):
        super().__init__()
        self.attn_l = MultiheadAttention(in_dim, num_heads=head_num)
        self.attn_max_num = attn_max_num

    def forward(self, img_feat, q_query):
        # for multi-head attn layer, input's shape should be as t, b, c
        img_feat = img_feat.view(*img_feat.shape[:2], -1).permute(2, 0, 1).contiguous()
        q_query = q_query.unsqueeze(0)
        attn_feat, _ = self.attn_l.forward(q_query, key=img_feat, value=img_feat, need_weights=False, attn_max_num=self.attn_max_num)
        return attn_feat.squeeze()


class HardAttnRnnLayer(HardAttnLayer):
    def __init__(self, img_dim, feat_dim, attn_max=40):
        super().__init__(attn_max)
        self.proj_l = Conv2D(img_dim, img_dim // 2, 1, orders=('coord', 'conv', 'act'))
        self.rnn_l = nn.GRU(img_dim//2, feat_dim, batch_first=True)

    def forward(self, x):
        x_top_inds = super().forward(x)
        x = self.proj_l(x)
        x = x.view(*x.shape[:2], -1)
        x = x.gather(-1, x_top_inds.unsqueeze(1).expand(*x.shape[:2], self.attn_max))  # b, c, max_num
        _, x_feat = self.rnn_l(x.permute(0, 2, 1))
        return x_feat.squeeze()




































