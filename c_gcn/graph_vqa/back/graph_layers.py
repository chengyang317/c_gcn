# coding=utf-8
from .layers import EdgeCondLayer, EdgeSpatialLayer, FusionCond, EdgeFeatLayer, \
    EdgeWeightLayer, GraphUpdateLayer, GraphClsFilm, NodeWeightLayer, NodeFeatLayer
from pt_pack.modules.layers.base_layers import Layer, Linear, Null
from .utils import build_graph_calc_layer
import torch
from .graph import Graph
import torch.nn as nn
from pt_pack.utils import str_split


__all__ = ['GraphLinearLayer', 'GraphSparseLayer', 'CalcGraphLayer', 'ClsGraphLayer']


class GraphLinearLayer(Layer):
    prefix = 'graph_linear_layer'

    def __init__(self, node_dim: int, out_dim: int, dropout: float=0.):
        super().__init__()
        self.proj_l = nn.Sequential(
            nn.Dropout(dropout),  # this is major change to ensure the feats in Node is complete.
            nn.utils.weight_norm(nn.Linear(node_dim, out_dim)),
            nn.ReLU(),
        )

    def forward(self, graph):
        coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
        node_feats = torch.cat((graph.node_feats, coord_feats), dim=-1)
        node_feats = self.proj_l(node_feats)  # b, k, hid_dim
        graph.node.update_feats(node_feats)
        return graph


class GraphSparseLayer(Layer):
    prefix = 'graph_sparse_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 edge_method: str,
                 weight_method: str,
                 graph_method: str,
                 dropout: float=0.,
                 ):
        super().__init__()
        self.edge_feat_l = EdgeFeatLayer(node_dim, cond_dim, edge_dim, edge_method, dropout)
        self.edge_weight_l = EdgeWeightLayer(edge_dim, weight_method, dropout)
        self.graph_update_l = GraphUpdateLayer(node_dim, cond_dim, out_dim, graph_method, dropout)

    def forward(self, graph):
        graph = self.edge_feat_l(graph)
        graph = self.edge_weight_l(graph)  # 2,m,k_size
        graph = self.graph_update_l(graph)
        return graph


class GraphConvLayer(Layer):
    prefix = 'graph_conv_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        edge_method, weight_method, param_method, node_method = str_split(method, '_')
        self.edge_feat_l = EdgeFeatLayer(node_dim, cond_dim, edge_dim, edge_method, dropout)
        self.edge_weight_l = EdgeWeightLayer(edge_dim, weight_method, dropout)
        self.edge_params_l = EdgePa
        self.node_feat_l = NodeFeatLayer(node_dim, cond_dim, out_dim, node_method, dropout)

    def forward(self, graph):
        graph = self.edge_feat_l(graph)
        graph = self.edge_weight_l(graph)  # 2,m,k_size
        graph = self.node_feat_l(graph)
        return graph




class ReduceGraphLayer(Layer):
    prefix = 'reduce_graph_layer'

    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 edge_method: str,
                 weight_method: str,
                 graph_method: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.edge_feat_l = EdgeFeatLayer(node_dim, cond_dim, edge_dim, edge_method, dropout)
        self.node_weight_l = NodeWeightLayer(edge_dim, weight_method, dropout)
        self.graph_update_l = GraphUpdateLayer(node_dim, cond_dim, out_dim, graph_method, dropout)

    def forward(self, graph):
        graph = self.edge_feat_l(graph)
        graph = self.node_weight_l(graph)  # 2,m,k_size
        graph = self.graph_update_l(graph)
        return graph


class ClsGraphLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        c_type, agg_method = method.split('_')
        if c_type == 'film':
            self.cls_l = GraphClsFilm(node_dim, cond_dim, out_dim, agg_method, dropout)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        return self.cls_l(graph)


class InitGraphLayer(Layer):
    prefix = 'init_graph_layer'

    def __init__(self, obj_dim: int, q_dim: int, out_dim: int, dropout: float=None):
        super().__init__()
        self.obj_proj_l = Linear(obj_dim, out_dim, orders=('linear', 'act'))
        self.edge_cond_l = FusionCond(obj_dim, q_dim, out_dim)
        if dropout is not None:
            self.drop_l = nn.Dropout(dropout)
        else:
            self.drop_l = Null()

    def forward(self, obj_feats, obj_coords, q_feats):
        node_feats = self.obj_proj_l(torch.cat(obj_feats, obj_coords), dim=-1)
        edge_feats = self.edge_cond_l(node_feats, q_feats)
        node_feats, edge_feats = self.drop_l(node_feats), self.drop_l(edge_feats)
        graph = Graph.build_from_node(node_feats, obj_coords, edge_feats, 'tri_u')
        return graph


class FusionGraphLayer(Layer):
    prefix = 'fusion_graph_layer'

    def __init__(self, obj_dim: int, q_dim: int, out_dim: int, cond_method: str, dropout: float=None):
        super().__init__()
        self.obj_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(obj_dim, out_dim)),
            nn.ReLU()
        )
        self.q_cond_l = FusionCond(out_dim, q_dim, out_dim)
        if dropout is not None:
            self.drop_l = nn.Dropout(dropout)
        else:
            self.drop_l = Null()

    def forward(self, graph, q_feats):
        coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
        node_feats = torch.cat((graph.node_feats, coord_feats), dim=-1)
        node_feats = self.obj_proj_l(node_feats)
        node_feats = self.q_cond_l(node_feats, q_feats)
        node_feats = self.drop_l(node_feats)
        graph.node.update(node_feats)
        return graph


class SpatialGraphLayer(Layer):
    prefix = 'spatial_graph_layer'

    def __init__(self,
                 obj_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 graph_method: str='kernel'
                 ):
        super().__init__()
        self.edge_weight_l = EdgeSpatialLayer(kernel_size)
        self.graph_calc_l = build_graph_calc_layer(graph_method, in_dim=obj_dim, out_dim=out_dim,
                                                   kernel_size=kernel_size)

    def forward(self, graph):
        graph = self.edge_weight_l(graph)  # 2,m,k_size
        graph = self.graph_calc_l(graph)
        return graph


class CondGraphLayer(Layer):
    prefix = 'cond_graph_layer'

    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 cond_method: str='sum_cond',
                 norm_method: str='instance',
                 graph_method: str='kernel',
                 use_pre_weight: bool=False
                 ):
        super().__init__()
        self.edge_weight_l = EdgeCondLayer(obj_dim, q_dim, cond_method, norm_method, kernel_size)
        self.graph_calc_l = build_graph_calc_layer(graph_method, in_dim=obj_dim, out_dim=out_dim,
                                                   kernel_size=kernel_size)
        self.use_pre_weight = use_pre_weight

    def forward(self, graph, q_feats):
        if self.use_pre_weight:
            assert graph.edge is not None
            edge_weight = graph.edge_weight
        graph = self.edge_weight_l(graph, q_feats)  # 2,m,k_size
        if self.use_pre_weight:
            edge_weight = edge_weight * graph.edge_weight
            graph.edge.edge_weight = edge_weight
        graph = self.graph_calc_l(graph)
        return graph




class CalcGraphLayer(Layer):
    prefix = 'calc_graph_layer'

    def __init__(self,
                 obj_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 graph_method: str='kernel',
                 ):
        super().__init__()
        self.graph_calc_l = build_graph_calc_layer(graph_method, in_dim=obj_dim, out_dim=out_dim,
                                                   kernel_size=kernel_size)

    def forward(self, graph):
        assert graph.edge is not None
        return self.graph_calc_l(graph)



























