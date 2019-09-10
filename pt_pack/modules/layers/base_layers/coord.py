# coding=utf-8
from .base import Layer
import torch
import torch.nn as nn


__all__ = ['Coord']


class ImgCoord(object):
    def __init__(self, name):
        self.name = name
        self.coord_buffer = {}

    def key(self, img_h, img_w):
        return f'{img_h}_{img_w}'

    def __getitem__(self, img_shape):
        batch_size, _, img_h, img_w = img_shape
        return self.get_coord(img_h, img_w, batch_size)

    def get_coord(self, img_h, img_w, batch_size=None):
        key = self.key(img_h, img_w)
        if key not in self.coord_buffer:
            self.coord_buffer[key] = coord = self.create_coord(img_h, img_w)
        else:
            coord = self.coord_buffer[key]
        if batch_size is None:
            return coord
        else:
            return coord.expand(batch_size, -1, -1, -1)

    def create_coord(self, img_h, img_w):
        h_coord = torch.linspace(-1, 1, steps=img_h).unsqueeze(1).expand(1, img_h, img_w)
        w_coord = torch.linspace(-1, 1, steps=img_w).unsqueeze(0).expand(1, img_h, img_w)
        return torch.cat((w_coord, h_coord))

    def __call__(self, x):
        return self[x.shape].type_as(x)

    # def create_coord(self, img_h, img_w):
    #     h_coord = torch.linspace(-1, 1, steps=img_h).unsqueeze(1).expand(1, img_h, img_w)
    #     w_coord = torch.linspace(-1, 1, steps=img_w).unsqueeze(0).expand(1, img_h, img_w)
    #     dist = torch.sqrt(h_coord**2 + w_coord**2)
    #     theta = torch.atan2(h_coord, w_coord)
    #     return torch.cat((dist, h_coord))

    # def create_coord(self, img_h, img_w):
    #     h_coord = torch.linspace(0, 1, steps=img_h).unsqueeze(1).expand(1, img_h, img_w)
    #     w_coord = torch.linspace(0, 1, steps=img_w).unsqueeze(0).expand(1, img_h, img_w)
    #     return torch.cat((w_coord, h_coord))


img_coord = ImgCoord('default')


class CoordLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 10, 3, padding=1),
            # nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_coord = img_coord(x)
        x_coord = self.layers(x_coord)
        return x_coord


class CoordLayerNew(nn.Module):
    def __init__(self, embed_num=196, embed_dim=10):
        super().__init__()
        self.embed_l = nn.Embedding(embed_num, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        b, c, h, w = x.shape
        x_coord = torch.linspace(0, h*w-1, h*w).unsqueeze(0).repeat(b, 1).type_as(x).long()
        x_coord = self.embed_l(x_coord).view(b, h, w, self.embed_dim).permute(0, 3, 1, 2)
        return x_coord


learned_layers = {}


class Coord(Layer):
    def __init__(self, type='default'):
        super().__init__()
        # assert type in ('learn', 'default')
        self.type = type
        if 'learn' in type:
            feat_h = int(type.split('-')[-1])
            global learned_layers
            if feat_h in learned_layers:
                self.layer = learned_layers[feat_h]
            else:
                self.layer = CoordLayerNew(embed_num=feat_h*feat_h)
                learned_layers[feat_h] = self.layer
        else:
            self.layer = img_coord

    @property
    def out_dim(self):
        return 2 if self.type == 'default' else 10

    def forward(self, x):
        out = torch.cat((x, self.layer(x)), dim=1)
        return out

    def build(cls, params, cls_name=None, sub_cls=None):
        return cls()