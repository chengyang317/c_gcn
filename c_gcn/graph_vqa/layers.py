# coding=utf-8
from . import modules as g_modules
from pt_pack import Layer, str_split
import torch
from .graph import Graph, Edge
import torch.nn as nn
import json


__all__ = ['GraphLinearLayer', 'GraphConvLayer', 'GraphClsLayer']


class GraphLinearLayer(Layer):
    prefix = 'graph_linear_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 param: str,
                 dropout: float = 0.):
        super().__init__()
        if param == 'linear':
            self.linear_l = nn.Sequential(
                # nn.Dropout(dropout),  # this is major change to ensure the feats in Node is complete.
                nn.utils.weight_norm(nn.Linear(node_dim, out_dim)),
                nn.ReLU(),
            )
        elif param == 'film':
            self.linear_l = g_modules.FilmFusion(node_dim, cond_dim, out_dim, dropout=0., cond_dropout=dropout)
        else:
            raise NotImplementedError()
        self.drop_l = nn.Dropout(dropout)
        self.method = param
        self.node_dim = node_dim

    def forward(self, graph):
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
            # coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            # node_feats = graph.node_feats
            # node_feats = torch.cat((node_feats, coord_feats), dim=-1)
            # node_feats = self.drop_l(node_feats)
        else:
            node_feats = self.drop_l(graph.node_feats)
        if self.method == 'linear':
            node_feats = self.linear_l(node_feats)  # b, k, hid_dim
        elif self.method == 'film':
            node_feats = self.linear_l(node_feats, graph.cond_feats)
        graph.node.update_feats(node_feats)
        return graph


class GraphConvLayer(Layer):
    prefix = 'graph_conv_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 param: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.params = json.loads(param)
        self.graph_learner_l = g_modules.GraphLearner(node_dim, cond_dim, edge_dim, self.params['edge'], dropout)
        self.graph_conv_l = g_modules.GraphConv(node_dim, cond_dim, edge_dim, out_dim, self.params['conv'], dropout)
        self.graph_pool_l = g_modules.GraphPool(self.params['pool'])

    def forward(self, graph: Graph):
        graph = self.graph_learner_l(graph)
        graph = self.graph_conv_l(graph)
        graph = self.graph_pool_l(graph)
        return graph


class GraphClsLayer(Layer):
    prefix = 'graph_cls_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 param: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        method, agg_method = param.split('_')
        if method == 'film':
            self.cls_l = g_modules.ClsFilm(node_dim, cond_dim, out_dim, agg_method, dropout)
        elif method == 'linear':
            self.cls_l = g_modules.ClsLinear(node_dim, out_dim, agg_method, dropout)
        elif method == 'rnn':
            self.cls_l = g_modules.ClsRnn(node_dim, cond_dim, out_dim, agg_method, dropout)
        elif method == 'cgs':
            self.cls_l = g_modules.ClsCgs(node_dim, cond_dim, out_dim, agg_method, dropout)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        return self.cls_l(graph)




































