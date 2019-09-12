# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from graph_vqa.graph_vqa import Graph
from pt_pack.modules.layers.base_layers import Layer, KerneLinear, Linear
from pt_pack.utils import node_intersect


__all__ = ['EdgeSpatialLayer', 'EdgeCondLayer', 'AggregateGraphCalcLayer', 'KernelGraphCalcLayer',
           'LinearGraphCalcLayer', 'GraphFilm', 'EdgeFeatLayer', 'EdgeWeightLayer', 'GraphUpdateLayer',
           'GraphClsFilm', 'NodeWeightLayer', 'NodeFeatLayer', 'EdgeParamLayer']


class EdgeSpatialLayer(Layer):
    def __init__(self,
                 kernel_size: int=8,
                 coord_method: str='polar'):
        super().__init__()
        self.coord_method = coord_method
        self.mean_rho = nn.Parameter(torch.Tensor(1, kernel_size))
        self.mean_theta = nn.Parameter(torch.Tensor(1, kernel_size))
        self.precision_rho = nn.Parameter(torch.Tensor(1, kernel_size))
        self.precision_theta = nn.Parameter(torch.Tensor(1, kernel_size))
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def calc_weight(self, graph):
        dist_coord = self.compute_pseudo(graph.node, graph.edge, self.coord_method)  # m,2
        diff = (dist_coord[:, 0].unsqueeze(-1) - self.mean_rho) ** 2  # m, k
        weights_rho = torch.exp(-0.5 * diff / (1e-14 + self.precision_rho ** 2))

        # compute theta weights
        first_angle = torch.abs(dist_coord[:, 1].unsqueeze(-1) - self.mean_theta)
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle) ** 2) / (1e-14 + self.precision_theta ** 2))
        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0
        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # m, k
        return weights

    def forward(self, graph):
        edge_weight = self.calc_weight(graph)
        graph.edge.weights = edge_weight
        return graph

    @staticmethod
    def compute_pseudo(node, edge, method):
        _, node_center = node.size_center
        node_center = node_center.view(-1, 2)
        coord = node_center[edge.coords[0]] - node_center[edge.coords[1]]  # m,2
        if method == 'cartesian':
            return coord
        elif method == 'polar':
            # Conver to polar coordinates
            coord_x, coord_y = coord.chunk(2, dim=-1)
            rho = torch.sqrt(coord_x**2 + coord_y**2)
            theta = torch.atan2(coord_x, coord_y)
            coord = torch.cat((rho, theta), dim=-1)
            return coord
        else:
            raise NotImplementedError()


class NodeCondFilm(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 n2n_method: str,
                 dropout: float=0.,
                 ):

        super().__init__()
        self.n2n_method = n2n_method
        self.node_proj_l = nn.Sequential(
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(node_dim, edge_dim)),
            nn.ReLU()
        )
        self.cond_proj_l = nn.Sequential(
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(cond_dim, edge_dim * 2)),
        )
        n_fusion_dim = edge_dim * 2 if n2n_method == 'cat' else edge_dim
        self.film_l = Linear(n_fusion_dim, edge_dim, norm_type='layer', norm_affine=False,
                             orders=('linear', 'norm', 'cond', 'act'))

    def forward(self, graph: Graph):
        node_feats = self.node_proj_l(graph.node_feats)
        batch_num, node_num, _ = node_feats.shape
        gamma, beta = self.cond_proj_l(graph.cond_feats).chunk(2, dim=-1)
        gamma += 1.
        n_fusion_feats = node_intersect(node_feats, method=self.n2n_method).view(batch_num * node_num * node_num, -1)
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            n_fusion_feats = n_fusion_feats[graph.edge_indexes]
        edge_num, o_c = n_fusion_feats.shape
        edge_feats = self.film_l(n_fusion_feats.view(batch_num, -1, o_c), gamma, beta)
        return edge_feats.view(edge_num, -1)


class EdgeQMul(nn.Module):
    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 kernel_size: int,
                 obj_method: str,
                 ):

        super().__init__()
        self.obj_method = obj_method
        self.obj_prob_l = nn.utils.weight_norm(nn.Linear(obj_dim, q_dim//2))
        inter_dim = q_dim // 2 * 2 if obj_method == 'cat' else q_dim // 2
        self.q_prob_l = nn.utils.weight_norm(nn.Linear(q_dim, inter_dim))

        self.logit_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(inter_dim, q_dim//4)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(q_dim//4, kernel_size))
        )

    def forward(self, graph, q_feats):
        if torch.isnan(q_feats).sum().item() > 0:
            print('debug')
        node_feats = self.obj_prob_l(graph.node_feats)
        b, obj_num, _ = node_feats.shape
        obj_feats = node_intersect(node_feats, method=self.obj_method).view(b * obj_num * obj_num, -1)
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            obj_feats = obj_feats[graph.indexes]
        m, o_c = obj_feats.shape  # m, o_c
        q_feats_re = self.q_prob_l(q_feats)[:, None, :].repeat(1, m // b, 1).view(m, -1)  # m, dim
        edge_feats = obj_feats * q_feats_re
        edge_logit = self.logit_l(edge_feats)
        return edge_logit


class EdgeQCat(nn.Module):

    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 kernel_size: int,
                 obj_method: str,
                 ):
        super().__init__()
        self.obj_method = obj_method
        self.obj_prob_l = Linear(obj_dim, q_dim//2, orders=('linear', 'act'))
        self.q_prob_l = Linear(q_dim, q_dim//2, orders=('linear', 'act'))
        self.logit_l = nn.Sequential(
            nn.Linear(self.calc_in_dim(q_dim//2, q_dim//2, obj_method), q_dim//4),
            nn.ReLU(inplace=True),
            nn.Linear(q_dim//4, kernel_size)
        )

    @staticmethod
    def calc_in_dim(obj_dim, q_dim, obj_method):
        if obj_method == 'cat':
            in_dim = obj_dim * 2 + q_dim
        elif obj_method == 'sum':
            in_dim = obj_dim + q_dim
        else:
            raise NotImplementedError()
        return in_dim

    def forward(self, graph, q_feats):
        node_feats = self.obj_prob_l(graph.node_feats)
        b, obj_num, _ = node_feats.shape
        q_feats = self.q_prob_l(q_feats)[:, None, :].repeat(1, obj_num*obj_num, 1).view(b*obj_num*obj_num, -1)
        obj_inter_feats = node_intersect(node_feats, method=self.obj_method).view(b * obj_num * obj_num, -1)
        edge_feats = torch.cat((obj_inter_feats, q_feats), dim=-1)
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            edge_feats = edge_feats[graph.indexes]
        edge_logit = self.logit_l(edge_feats)  # b*e_size, k_size
        return edge_logit


class EdgeQProj(nn.Module):

    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 kernel_size: int,
                 obj_method: str,
                 ):
        super().__init__()
        hid_dim = q_dim //2
        self.obj_method = obj_method
        self.obj_prob_l = Linear(obj_dim, hid_dim, orders=('linear', 'act'))
        self.q_prob_l = Linear(q_dim, hid_dim*kernel_size, orders=('linear', 'act'))
        self.kernel_size = kernel_size

    @staticmethod
    def calc_in_dim(obj_dim, q_dim, obj_method):
        if obj_method == 'cat':
            in_dim = obj_dim * 2 + q_dim
        elif obj_method == 'sum':
            in_dim = obj_dim + q_dim
        else:
            raise NotImplementedError()
        return in_dim

    def forward(self, graph, q_feats):
        node_feats = self.obj_prob_l(graph.node_feats)
        b, obj_num, _ = node_feats.shape
        obj_feats = node_intersect(node_feats, method=self.obj_method).view(b * obj_num * obj_num, -1)
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            obj_feats = obj_feats[graph.indexes]
        m, o_c = obj_feats.shape  # m, o_c
        q_feats = self.q_prob_l(q_feats)[:, None, :].repeat(1, m//b, 1).view(m, -1, self.kernel_size)  # m, dim, k
        edge_logit = obj_feats[:, None, :].bmm(q_feats).squeeze()  # m, k
        edge_logit = edge_logit / q_feats.norm(dim=1)
        return edge_logit


class EdgeQSim(nn.Module):

    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 kernel_size: int,
                 obj_method: str,
                 ):
        super().__init__()
        self.proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(obj_dim+q_dim, q_dim)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(q_dim, q_dim)),
            nn.ReLU(inplace=True),
        )
        self.kernel_size = kernel_size
        assert kernel_size == 1

    @staticmethod
    def calc_in_dim(obj_dim, q_dim, obj_method):
        if obj_method == 'cat':
            in_dim = obj_dim * 2 + q_dim
        elif obj_method == 'sum':
            in_dim = obj_dim + q_dim
        else:
            raise NotImplementedError()
        return in_dim

    def forward(self, graph, q_feats):
        feats = torch.cat((graph.node_feats, q_feats[:, None, :].repeat(1, graph.node_num, 1)), dim=-1)
        feats = self.proj_l(feats)
        edge_logit = torch.matmul(feats, feats.transpose(1, 2)).view(-1, 1)  # b*obj_num*obj_num, 1
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            edge_logit = edge_logit[graph.indexes]
        return edge_logit


class EdgeLogitLayer(nn.Module):
    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 kernel_size: int,
                 cond_method: str,
                 ):
        super().__init__()
        obj_method, obj_q_method = cond_method.split('_')
        if obj_q_method == 'cat':
            self.logit_l = EdgeQCat(obj_dim, q_dim, kernel_size, obj_method)
        elif obj_q_method == 'cond':
            self.logit_l = NodeCondFilm(obj_dim, q_dim, kernel_size, obj_method)
        elif obj_q_method == 'proj':
            self.logit_l = EdgeQProj(obj_dim, q_dim, kernel_size, obj_method)
        elif obj_q_method == 'mul':
            self.logit_l = EdgeQMul(obj_dim, q_dim, kernel_size, obj_method)
        elif obj_q_method == 'sim':
            self.logit_l = EdgeQSim(obj_dim, q_dim, kernel_size, obj_method)
        else:
            raise NotImplementedError()

    def forward(self, graph, q_feats):
        graph.edge.remove_self_loop()
        edge_logit = self.logit_l(graph, q_feats)  # m, k_size
        edge_logit = graph.edge.to_mirror(edge_logit)
        return edge_logit



class EdgeCondLayer(nn.Module):
    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 cond_method: str,
                 norm_method: str,
                 kernel_size: int=1,
                 reduce_size: int=None,
                 ):
        super().__init__()
        self.logit_l = EdgeLogitLayer(obj_dim, q_dim, kernel_size, cond_method)
        self.norm_method = norm_method
        self.reduce_size = reduce_size

    def forward(self, graph: Graph, q_feats):
        edge_logit = self.logit_l(graph, q_feats)  # m, k_size
        node_weights = graph.edge.logit2weight(edge_logit, self.norm_method, self.reduce_size)
        graph.node.update(node_weights=node_weights)
        return graph


class NodeCondLayer(nn.Module):
    def __init__(self,
                 obj_dim: int,
                 q_dim: int,
                 cond_method: str,
                 norm_method: str,
                 kernel_size: int=1,
                 reduce_size: int=None,
                 ):
        super().__init__()
        self.logit_l = EdgeLogitLayer(obj_dim, q_dim, kernel_size, cond_method)
        self.norm_method = norm_method
        self.reduce_size = reduce_size

    def forward(self, graph, q_feats):
        edge_logit = self.logit_l(graph, q_feats)  # m, k_size
        graph.node.logit2weight(edge_logit, self.norm_method, self.reduce_size)
        if self.reduce_size is not None:
            graph.init_edge()
        return graph


class AggregateGraphCalcLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, q_feats):
        graph.aggregate(graph.node_feats)
        return graph


class KernelGraphCalcLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 kernel_size: int,
                 ):
        super().__init__()
        self.kernel_linear_l = KerneLinear(in_dim, out_dim // kernel_size, kernel_size)
        self.relu_l = nn.ReLU()
        self.kernel_size = kernel_size

    def forward(self, graph: Graph, q_feats=None):
        b_size, obj_num, _ = graph.node.shape
        node_feats = self.relu_l(self.kernel_linear_l(graph.node_feats)).view(b_size, obj_num, self.kernel_size, -1)
        graph.aggregate(node_feats)
        return graph





class LinearGraphCalcLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 ):
        super().__init__()
        self.layers = nn.Sequential(
            Linear(in_dim, out_dim//2, orders=('linear', 'act')),
            Linear(out_dim//2, out_dim, orders=('linear', 'act'))
        )

    def forward(self, graph: Graph, q_feats=None):
        b_size, obj_num, _ = graph.node.shape
        node_feats = self.layers(graph.node_feats).view(b_size, obj_num, -1)
        graph.aggregate(node_feats, method='mean')
        return graph


class FusionMul(nn.Module):
    def __init__(self, obj_dim: int, q_dim: int, out_dim: int):
        super().__init__()
        self.linear_0 = nn.utils.weight_norm(nn.Linear(obj_dim, out_dim))
        self.linear_1 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))
        self.relu_l = nn.ReLU()

    def forward(self, obj_feats, q_feats):
        feat = obj_feats * self.relu_l(q_feats)
        feat = self.linear_0(feat)
        logits = self.linear_1(feat)
        return logits


class FusionCond(nn.Module):
    def __init__(self, obj_dim: int, q_dim: int, out_dim: int):
        super().__init__()
        self.q_proj_l = nn.utils.weight_norm(nn.Linear(q_dim, out_dim//2 * 2))
        self.cond_l = Linear(obj_dim, out_dim // 2, norm_type='layer', norm_affine=False,
                             orders=('linear', 'norm', 'cond', 'act'))
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim//2)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(out_dim//2, out_dim))
        )

    def forward(self, obj_feats, q_feats):
        gamma, beta = self.q_proj_l(q_feats).chunk(2, dim=-1)
        gamma += 1.
        cond_feats = self.cond_l(obj_feats, gamma, beta)
        feats = self.linear_l(cond_feats)
        return feats




class FilmFusion(nn.Module):
    def __init__(self,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 norm_type='layer',
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.cond_proj_l = nn.Sequential(
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(cond_dim, out_dim*2))
        )
        self.drop_l = nn.Dropout(dropout)
        self.film_l = Linear(in_dim, out_dim // 2, norm_type=norm_type, norm_affine=False,
                             orders=('linear', 'norm', 'cond', 'act'))

    def forward(self, x, cond):
        gamma, beta = self.cond_proj_l(cond).chunk(2, dim=-1)
        gamma += 1.
        x = self.drop_l(x)
        x = self.film_l(x, gamma, beta)
        return x


class GraphFilm(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str = 'mean',
                 dropout: float = 0.
                 ):
        super().__init__()
        self.cond_proj_l = nn.Sequential(
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(cond_dim, out_dim))
        )
        self.drop_l = nn.Dropout(dropout)
        self.film_l = Linear(node_dim, out_dim // 2, norm_type='layer', norm_affine=False,
                             orders=('linear', 'norm', 'cond', 'act'))
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim)),
        )
        self.relu_l = nn.ReLU()
        self.method = method

    def forward(self, graph: Graph):
        gamma, beta = self.cond_proj_l(graph.cond_feats).chunk(2, dim=-1)
        gamma += 1.
        node_feats = self.drop_l(graph.node_feats)
        node_feats = self.film_l(node_feats, gamma, beta)
        node_feats = self.linear_l(node_feats)
        graph.node.update_feats(node_feats)
        if self.method != 'none':
            graph.aggregate(method=self.method)
        graph.node.update_feats(self.relu_l(graph.node_feats))
        return graph


class GraphClsFilm(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float=0.
                 ):
        super().__init__()
        self.cond_proj_l = nn.Sequential(
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Linear(cond_dim, out_dim // 2 * 2)),
        )
        self.drop_l = nn.Dropout(dropout)
        self.cond_l = Linear(node_dim, out_dim // 2, norm_type='layer', norm_affine=False,
                             orders=('linear', 'norm', 'cond', 'act'))
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim // 2)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim))
        )
        self.method = method

    def forward(self, graph: Graph):
        gamma, beta = self.cond_proj_l(graph.cond_feats).chunk(2, dim=-1)
        gamma += 1.
        node_feats = self.drop_l(graph.pooling_feats(self.method))
        feats = self.cond_l(node_feats, gamma, beta)
        logits = self.linear_l(feats)
        return logits


class EdgeFeatLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 method: str,
                 dropout: float=0.
                 ):
        super().__init__()
        if method == 'share':
            self.feat_l = None
        else:
            n2n_method, n2c_method = method.split('_')
            if n2c_method == 'cat':
                self.feat_l = EdgeQCat(node_dim, cond_dim, edge_dim, n2n_method)
            elif n2c_method == 'film':
                self.feat_l = NodeCondFilm(node_dim, cond_dim, edge_dim, n2n_method, dropout)
            elif n2c_method == 'proj':
                self.feat_l = EdgeQProj(node_dim, cond_dim, edge_dim, n2n_method)
            elif n2c_method == 'mul':
                self.feat_l = EdgeQMul(node_dim, cond_dim, edge_dim, n2n_method)
            elif n2c_method == 'sim':
                self.feat_l = EdgeQSim(node_dim, cond_dim, edge_dim, n2n_method)
            else:
                raise NotImplementedError()
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.edge_dim = edge_dim
        self.edge_method = method

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}_{self.edge_dim}'

    def forward(self, graph: Graph):
        graph.edge.remove_self_loop()
        feat_l = self.feat_l or graph.edge.feat_layers[self.layer_key]
        edge_feats = feat_l(graph)  # m, c
        if graph.edge.feat_layers.get(self.layer_key, None) is None:
            graph.edge.feat_layers[self.layer_key] = self.feat_l
        graph.edge.feats = edge_feats
        return graph


class EdgeParamLayer(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        if method == 'share':
            self.param_l = None
        else:
            self.param_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, out_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(out_dim//2, out_dim))
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}_{self.out_dim}'

    def forward(self, graph: Graph):
        param_l = self.param_l or graph.edge.param_layers[self.layer_key]
        params = param_l(graph.edge.feats)
        graph.edge.params = params
        if graph.edge.param_layers.get(self.layer_key, None) is None:
            graph.edge.param_layers[self.layer_key] = self.param_l
        return graph


class NodeFeatLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        if method == 'share':
            pass
        self.edge_para_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(edge_dim, out_dim)),
            nn.Tanh(),
        )
        self.node_update_l = FilmFusion(node_dim, cond_dim, out_dim, dropout=dropout)

    def forward(self, graph: Graph):
        edge_params = self.edge_para_l(graph.edge_feats)
        node_feats = self.node_update_l(graph.node_feats)
        edge_mirror = graph.edge.load_structure
        edge_params = edge_mirror.to_mirror(edge_params) * edge_mirror.weights
        node_feats = edge_mirror.aggregate(node_feats, edge_params)
        graph.node.update_feats(node_feats)


class EdgeWeightLayer(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.norm_method, kernel_size, reduce_size = method.split('_')
        self.reduce_size = int(reduce_size)
        if kernel_size == 'share':
            self.logit_l = None
        else:
            self.logit_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, edge_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(edge_dim//2, int(kernel_size)))
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}'

    def forward(self, graph: Graph):
        logit_l = self.logit_l or graph.edge.logit_layers[self.layer_key]
        logit = logit_l(graph.edge.feats)
        graph.edge.logits = logit
        if graph.edge.logit_layers.get(self.layer_key, None) is None:
            graph.edge.logit_layers[self.layer_key] = self.logit_l
        graph.edge.logit2weight(self.norm_method, self.reduce_size)
        return graph


class NodeWeightLayer(nn.Module):
    def __init__(self,
                 edge_dim: int,
                 method: str,
                 dropout: float=0.
                 ):
        super().__init__()
        self.edge_dim = edge_dim
        self.norm_method, kernel_size, reduce_size = method.split('_')
        self.reduce_size = int(reduce_size)
        if kernel_size == 'share':
            self.edge_logit_l = None
        else:
            self.edge_logit_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, edge_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(edge_dim//2, int(kernel_size)))
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}'

    def forward(self, graph: Graph):
        edge_logit_l = self.edge_logit_l or graph.edge.logit_layers[self.layer_key]
        edge_logit = edge_logit_l(graph.edge.feats)
        if graph.edge.logit_layers.get(self.layer_key, None) is None:
            graph.edge.logit_layers[self.layer_key] = self.edge_logit_l
        graph.node.logit2weight(self.norm_method, self.reduce_size)
        return graph


class GraphUpdateLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float=0.,
                 ):
        super().__init__()
        self.method = method
        g_type, agg_method = method.split('_')
        if g_type == 'film':
            self.update_l = GraphFilm(node_dim, cond_dim, out_dim, agg_method, dropout=dropout)
        else:
            raise NotImplementedError()

    def forward(self, graph):
        return self.update_l(graph)









































