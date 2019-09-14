# coding=utf-8
from . import modules as g_modules
from pt_pack import Layer, str_split
import torch
from .graph import Graph, Edge
import torch.nn as nn
import json


__all__ = ['GraphLinearLayer', 'CondGraphConvLayer', 'CondGraphClsLayer']


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
            coord_feats = torch.cat(graph.node.size_center, dim=-1)
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



class CondGraphConvLayer(Layer):
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
        self.graph_learner_l = g_modules.CondGraphLearner(node_dim, cond_dim, edge_dim, self.params['edge'], dropout)
        self.graph_conv_l = g_modules.CondGraphConv(node_dim, cond_dim, edge_dim, out_dim, self.params['conv'], dropout)

    def forward(self, graph: Graph):
        graph = self.graph_learner_l(graph)
        graph = self.graph_conv_l(graph)
        # graph.calc_node_weights()
        graph.feats.append(graph.pooling_feats(self.params['pool']))
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
        """

        :param node_dim:
        :param cond_dim:
        :param edge_dim:
        :param out_dim:
        :param param: cgs:x_cond:x_x
        :param dropout:
        """
        super().__init__()
        self.graph_learner_l, graph_conv_l = None, None
        edge_method, conv_method, self.pool_method = str_split(param, '_')
        edge_method, method_param = str_split(edge_method, ':')
        if edge_method == 'cgs':
            self.graph_learner_l = g_modules.CgsGraphLearner(node_dim, cond_dim, method_param, 512, dropout)
        elif edge_method == 'cond':
            self.graph_learner_l = g_modules.CondGraphLearner(node_dim, cond_dim, edge_dim, method_param, dropout)
        else:
            raise NotImplementedError()
        conv_method, method_param = str_split(conv_method, ':')
        if conv_method == 'cgs':
            self.graph_conv_l = nn.Sequential(
                g_modules.CgsGraphConv(node_dim, out_dim * 2, method_param, dropout=dropout),
                g_modules.CgsGraphConv(out_dim * 2, out_dim, method_param, use_graph_weights=False, dropout=dropout)
            )
        elif conv_method == 'cond':
            self.graph_conv_l = g_modules.CondGraphConv(node_dim, cond_dim, edge_dim, out_dim, method_param, dropout)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        graph = self.graph_learner_l(graph)
        graph = self.graph_conv_l(graph)
        graph.feats.append(graph.pooling_feats(self.pool_method))
        return graph


class CgsGraphConvLayer(Layer):
    """
    """
    prefix = 'graph_conv_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 params: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        weight_method, conv_method = str_split(params, '_')
        self.edge_learner_l = g_modules.CgsGraphLearner(node_dim, cond_dim, weight_method, hid_dim=512)
        self.conv_layers = nn.ModuleList([
            g_modules.CgsGraphConv(node_dim, out_dim * 2, conv_method),
            g_modules.CgsGraphConv(out_dim * 2, out_dim, conv_method, use_graph_weights=False)
        ])

    def forward(self, graph: Graph):
        graph = self.edge_learner_l(graph)
        for conv_l in self.conv_layers:
            graph = conv_l(graph)
        graph.feats.append(graph.pooling_feats('max'))
        return graph


class CondGraphPoolLayer(Layer):
    prefix = 'graph_pool_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 params: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        weight_params, reduce_size = str_split(params, '_')
        self.node_weight_l = g_modules.NodeWeightLayer(node_dim, edge_dim, weight_params, dropout)
        self.node_pool_l = g_modules.NodePoolLayer(reduce_size)

    def forward(self, graph: Graph):
        graph = self.node_weight_l(graph)
        graph = self.node_pool_l(graph)
        return graph


# class CondGraphPoolLayer(Layer):
#     prefix = 'graph_pool_layer'
#
#     def __init__(self,
#                  node_dim: int,
#                  cond_dim: int,
#                  edge_dim: int,
#                  method: str,
#                  dropout: float = 0.,
#                  ):
#         super().__init__()
#         edge_method, weight_method = str_split(method, '-')
#         self.edge_feat_l = g_modules.EdgeFeatLayer(node_dim, cond_dim, edge_dim, edge_method, dropout)
#         self.node_weight_l = g_modules.NodeWeightLayer(node_dim-4, edge_dim, weight_method, dropout)
#
#     def forward(self, graph: Graph):
#         graph = self.edge_feat_l(graph)
#         graph = self.node_weight_l(graph)  # 2,m,k_size
#         new_edge = Edge(graph.node, graph.edge.method)
#         new_edge.feat_layers = graph.edge.feat_layers
#         new_edge.logit_layers = graph.edge.logit_layers
#         new_edge.param_layers = graph.edge.param_layers
#         graph.edge = new_edge
#         return graph


class CondGraphClsLayer(Layer):
    prefix = 'graph_cls_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 param: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        linear_method, agg_method = param.split('_')
        if linear_method == 'film':
            self.cls_l = g_modules.GraphClsFilm(node_dim, cond_dim, out_dim, agg_method, dropout)
        elif linear_method == 'linear':
            self.cls_l = g_modules.GraphClsLinear(node_dim, out_dim, agg_method, dropout)
        elif linear_method == 'rnn':
            self.cls_l = g_modules.GraphClsRnn(node_dim, cond_dim, out_dim, agg_method, dropout)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        return self.cls_l(graph)


class CgsGraphClsLayer(Layer):
    prefix = 'graph_cls_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 params: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.agg_method = params
        self.logit_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(node_dim, out_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(out_dim, out_dim)),
        )
        self.relu_l = nn.ReLU()

    def forward(self, graph: Graph):
        feats = graph.graph_feats(self.agg_method)
        feats = self.relu_l(graph.cond_feats) * feats
        logits = self.logit_l(feats)
        return logits

































