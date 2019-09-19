# coding=utf-8
import torch
import torch.nn as nn
from .graph import Graph, EdgeAttr, EdgeNull, EdgeAddLoop, EdgeMirror, EdgeTopK, Edge
from pt_pack import Linear, Null, node_intersect, str_split, try_set_attr
import numpy as np


__all__ = ['EdgeFeatLayer', 'EdgeWeightLayer', 'NodeWeightLayer', 'NodeFeatLayer', 'EdgeParamLayer']


class FilmFusion(nn.Module):
    def __init__(self,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 norm_type='layer',
                 dropout: float = 0.,
                 act_type: str = 'relu'
                 ):
        super().__init__()
        self.cond_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(cond_dim, out_dim*2)),
            # nn.ReLU()
        )
        self.drop_l = nn.Dropout(dropout)
        self.film_l = Linear(in_dim, out_dim, norm_type=norm_type, norm_affine=False,
                             orders=('linear', 'norm', 'cond'))
        if act_type is 'relu':
            self.relu_l = nn.ReLU()
        elif act_type is 'none':
            self.relu_l = Null()
        elif act_type is 'tanh':
            self.relu_l = nn.Tanh()
        else:
            raise NotImplementedError()

    def forward(self, x, cond):
        gamma, beta = self.cond_proj_l(cond).chunk(2, dim=-1)
        gamma += 1.
        x = self.drop_l(x)
        x = self.film_l(x, gamma, beta)
        x = self.relu_l(x)
        return x


class EdgeFeatFilm(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 n2n_method: str,
                 dropout: float = 0.,
                 ):

        super().__init__()
        self.n2n_method = n2n_method
        self.node_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(node_dim, edge_dim)),
            nn.ReLU()
        )
        edge_in_dim = edge_dim+8 if n2n_method in ('sum', 'mul', 'max') else edge_dim*2+8
        # self.edge_proj_l = nn.Sequential(
        #     nn.utils.weight_norm(nn.Linear(edge_in_dim, edge_dim)),
        #     nn.ReLU()
        # )
        self.drop_l = nn.Dropout(dropout)
        self.film_l = FilmFusion(edge_in_dim, cond_dim, edge_dim, act_type='relu')

        self.node_dim = node_dim

    def forward(self, graph: Graph):
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
        else:
            node_feats = self.drop_l(graph.node_feats)
        node_feats = self.node_proj_l(node_feats)
        batch_num, node_num, _ = node_feats.shape

        joint_feats = node_intersect(node_feats, method=self.n2n_method)
        joint_feats = torch.cat((joint_feats, graph.edge.spatial_feats()), dim=-1).view(batch_num*node_num*node_num, -1)

        joint_feats = graph.edge.attr_process(joint_feats)
        # joint_feats = self.edge_proj_l(joint_feats)
        edge_num, o_c = joint_feats.shape

        edge_feats = self.film_l(joint_feats.view(batch_num, -1, o_c), graph.cond_feats)
        return edge_feats.view(edge_num, -1)

    def compute_pseudo(self, graph: Graph):
        node_size, node_centre = graph.node.spatial_attr
        node_dis = node_intersect(node_centre, 'minus')  # b, k, k, 2
        node_dis = node_dis.view(-1, 2)
        coord_x, coord_y = node_dis.chunk(2, dim=-1)
        rho = torch.sqrt(coord_x ** 2 + coord_y ** 2)
        theta = torch.atan2(coord_x, coord_y)
        coord = torch.cat((rho, theta), dim=-1)  # m, 2
        return coord


class EdgeFeatMul(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 n2n_method: str,
                 dropout: float = 0.
                 ):

        super().__init__()
        self.n2n_method = n2n_method
        self.node_dim = node_dim
        self.node_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(node_dim, edge_dim)),
            nn.ReLU()
        )
        self.drop_l = nn.Dropout(dropout)
        self.joint_proj_l = nn.Sequential(
            nn.utils.weight_norm((nn.Linear(edge_dim*2 + 8 if n2n_method == 'cat' else edge_dim+8, edge_dim))),
            nn.ReLU()
        )
        self.q_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(cond_dim, edge_dim)),
            nn.ReLU()
        )
        self.relu_l = nn.ReLU()

    def forward(self, graph: Graph):
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
        else:
            node_feats = self.drop_l(graph.node_feats)
        node_feats = self.node_proj_l(node_feats)
        batch_num, node_num, _ = node_feats.shape

        joint_feats = node_intersect(node_feats, method=self.n2n_method)
        joint_feats = torch.cat((joint_feats, graph.edge.spatial_feats()), dim=-1).view(batch_num*node_num*node_num, -1)
        joint_feats = graph.edge.attr_process(joint_feats)
        joint_feats = self.joint_proj_l(joint_feats)
        edge_num, o_c = joint_feats.shape

        edge_feats = joint_feats.view(batch_num, -1, o_c) * self.q_proj_l(graph.cond_feats).unsqueeze(1)
        return self.relu_l(edge_feats.view(edge_num, -1))


class EdgeFeatCat(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 n2n_method: str,
                 dropout: float = 0.
                 ):

        super().__init__()
        self.n2n_method = n2n_method
        self.node_dim = node_dim
        self.node_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(node_dim, edge_dim)),
            nn.ReLU()
        )
        self.drop_l = nn.Dropout(dropout)
        self.joint_proj_l = nn.Sequential(
            nn.utils.weight_norm((nn.Linear(edge_dim*2 if n2n_method == 'cat' else edge_dim, edge_dim))),
            nn.ReLU()
        )
        self.q_proj_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(cond_dim, edge_dim)),
            nn.ReLU()
        )
        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(edge_dim*2, edge_dim)),
            nn.ReLU()
        )

    def forward(self, graph: Graph):
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
        else:
            node_feats = self.drop_l(graph.node_feats)
        node_feats = self.node_proj_l(node_feats)
        batch_num, node_num, _ = node_feats.shape

        n_fusion_feats = node_intersect(node_feats, method=self.n2n_method).view(batch_num * node_num * node_num, -1)
        if graph.edge is not None:
            graph.edge.remove_self_loop()
            n_fusion_feats = n_fusion_feats[graph.edge_indexes]
        joint_feats = self.joint_proj_l(n_fusion_feats)
        edge_num, o_c = joint_feats.shape
        q_feats = self.q_proj_l(graph.cond_feats).unsqueeze(1).expand(-1, edge_num//batch_num, -1)
        joint_feats = torch.cat((joint_feats.view(batch_num, -1, o_c), q_feats), dim=-1)
        edge_feats = self.linear_l(joint_feats)
        return edge_feats.view(edge_num, -1)


class ClsFilm(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.film_l = FilmFusion(node_dim, cond_dim, out_dim//2, dropout=dropout)

        self.linear_l = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim // 2)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim))
        )
        self.method = method

    def forward(self, graph: Graph):
        graph_feats = graph.graph_feats(self.method)
        feats = self.film_l(graph_feats, graph.cond_feats)
        logits = self.linear_l(feats)
        return logits


class ClsLinear(nn.Module):
    def __init__(self,
                 node_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.linear_l = nn.Sequential(
            nn.Dropout(dropout),
            # nn.utils.weight_norm(nn.Linear(node_dim, out_dim)),
            # nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(node_dim, out_dim // 2)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(out_dim // 2, out_dim))
        )
        self.method = method

    def forward(self, graph: Graph):
        graph_feats = graph.graph_feats(self.method)
        # feats = graph_feats + feats  # major change
        logits = self.linear_l(graph_feats)
        return logits


class ClsRnn(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.q_proj_l = nn.Sequential(
            nn.Linear(cond_dim, out_dim),
            nn.Tanh()
        )
        self.rnn_l = nn.GRU(node_dim, out_dim, batch_first=True, dropout=dropout, num_layers=1)
        self.linear_l = nn.Linear(out_dim, out_dim)
        self.method = method

    def forward(self, graph: Graph):
        graph_feats = torch.stack(graph.layer_feats, dim=1)
        outs, feats = self.rnn_l(graph_feats, self.q_proj_l(graph.cond_feats).unsqueeze(0))
        logits = self.linear_l(feats.squeeze())
        return logits


class ClsCgs(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.agg_method = method
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


class CondNodeFeat(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.,
                 use_gin: bool = False,
                 ):
        super().__init__()
        self.feat_method, self.agg_method = str_split(method, '^')
        if self.feat_method == 'share':
            self.feat_l = None
        elif self.feat_method == 'film':
            self.feat_l = FilmFusion(node_dim, cond_dim, out_dim, act_type='relu')
        elif self.feat_method == 'linear':
            self.feat_l = nn.Sequential(
                Linear(node_dim, out_dim // 2, orders=('linear', 'act')),
                Linear(out_dim // 2, out_dim, orders=('linear', 'act'))
            )
        else:
            raise NotImplementedError()
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.drop_l = nn.Dropout(dropout)
        self.act_l = nn.ReLU()
        self.use_gin = use_gin
        self.norm_l = lambda x: x
        # self.norm_l = nn.LayerNorm([out_dim])
        # self.norm_l = nn.InstanceNorm1d(out_dim)
        # self.norm_l = nn.LayerNorm([36, out_dim])
        # self.norm_l = nn.BatchNorm1d(out_dim)
        if self.use_gin:
            self.eps = torch.nn.Parameter(torch.zeros(out_dim))
        # self.linear_l = nn.utils.weight_norm(nn.Linear(out_dim + out_dim, out_dim))

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}'

    def forward(self, graph: Graph):
        feat_l = self.feat_l or graph.node.feat_layers[self.layer_key]
        if self.layer_key not in graph.node.feat_layers:
            graph.node.feat_layers[self.layer_key] = feat_l

        origin_node_feats = graph.node_feats
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
        else:
            node_feats = self.drop_l(graph.node_feats)
        if feat_l._get_name() == 'FilmFusion':
            new_node_feats = feat_l(node_feats, graph.cond_feats)
        else:
            new_node_feats = feat_l(node_feats)

        edge_weights = graph.edge_attrs['weights'].value * graph.edge_attrs['params'].value
        last_op = graph.edge_attrs['weights'].op

        if last_op.is_loop:
            if self.use_gin:
                edge_weights[last_op.loop_mask()] += 1.0
            # if self.use_gin:
            #     edge_weights[last_op.loop_mask()] = 1.0 + self.eps

        b_num, n_num, c_num = new_node_feats.shape
        edge_weights = edge_weights.view(b_num, n_num, -1, edge_weights.shape[-1])
        node_j_feats = new_node_feats.view(b_num * n_num, c_num)[last_op.coords[1]].view(b_num, n_num, -1, c_num)
        nb_feats = edge_weights * node_j_feats
        nb_feats = nb_feats.sum(dim=2)

        if not last_op.is_loop:
            if self.use_gin:
                node_feats = node_feats * (1. + self.eps[None, None, :])
            node_feats = new_node_feats + nb_feats
        else:
            # node_feats = torch.cat((new_node_feats, nb_feats), dim=-1)
            # node_feats = self.linear_l(node_feats)
            node_feats = nb_feats
            # node_feats = origin_node_feats + nb_feats

        # node_feats = self.act_l(self.norm_l(node_feats.view(-1, self.out_dim, 1))).view(b_num, n_num, self.out_dim)
        node_feats = self.act_l(self.norm_l(node_feats))
        # node_feats = self.act_l(self.norm_l(node_feats.transpose(1, 2))).transpose(1, 2)
        graph.node.update_feats(node_feats)
        return graph


class CgsNodeFeat(nn.Module):
    def __init__(self,
                 node_dim: int,
                 out_dim: int,
                 method: str,
                 use_graph_weights: bool = True,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.node_dim = node_dim
        self.agg_method, kernel_size = str_split(method, '^')
        self.mean_rho = nn.Parameter(torch.Tensor(1, kernel_size))
        self.mean_theta = nn.Parameter(torch.Tensor(1, kernel_size))
        self.precision_rho = nn.Parameter(torch.Tensor(1, kernel_size))
        self.precision_theta = nn.Parameter(torch.Tensor(1, kernel_size))

        self.conv_layers = nn.ModuleList(
            [nn.Linear(node_dim, out_dim // kernel_size, bias=False) for _ in range(kernel_size)]
        )
        self.relu_l = nn.ReLU()
        self.use_graph_weights = use_graph_weights
        self.drop_l = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

    def compute_pseudo(self, graph: Graph):
        node_size, node_centre = graph.node.spatial_attr
        node_dis = node_intersect(node_centre, 'minus')  # b, k, k, 2
        node_dis = node_dis.view(-1, 2)
        node_dis = graph.edge.topk_op().attr_process(EdgeAttr('node_dist', node_dis, EdgeNull()))

        coord_x, coord_y = node_dis.value.chunk(2, dim=-1)
        rho = torch.sqrt(coord_x ** 2 + coord_y ** 2)
        theta = torch.atan2(coord_x, coord_y)
        coord = torch.cat((rho, theta), dim=-1)  # m, 2
        return coord

    def gaussian_weight(self, graph: Graph):
        dist_coord = self.compute_pseudo(graph)

        diff = (dist_coord[:, 0].unsqueeze(-1) - self.mean_rho) ** 2  # m, k
        weights_rho = torch.exp(-0.5 * diff / (1e-14 + self.precision_rho ** 2))

        # compute theta weights
        first_angle = torch.abs(dist_coord[:, 1].unsqueeze(-1) - self.mean_theta)
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(
            -0.5 * (torch.min(first_angle, second_angle) ** 2) / (1e-14 + self.precision_theta ** 2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)  # m, k
        return EdgeAttr('gaussian_weights', weights, graph.edge.remove_loop_op())

    def forward(self, graph: Graph):
        if self.node_dim % 512 == 4:
            coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
            node_feats = self.drop_l(graph.node_feats)
            node_feats = torch.cat((node_feats, coord_feats), dim=-1)
        else:
            node_feats = self.drop_l(graph.node_feats)
        edge_dist_weights = self.gaussian_weight(graph)  # m, k

        if self.use_graph_weights:
            edge_weights = graph.edge.edge_attrs['weights'].value * edge_dist_weights.value  # n, k
        else:
            edge_weights = edge_dist_weights.value

        topk_op = graph.edge.topk_op()
        assert topk_op.is_loop
        b_num, n_num, c_num = node_feats.shape
        edge_weights = edge_weights.view(b_num*n_num, -1, edge_weights.shape[-1]).transpose(1, 2)  # m, k_num, 16
        node_j_feats = node_feats.view(b_num * n_num, c_num)[topk_op.coords[1]].view(b_num*n_num, -1, c_num)  # m, 16, c

        node_feats = torch.bmm(edge_weights, node_j_feats).view(b_num, n_num, -1, c_num)  # b, n_num, k_num, c_num
        node_feats = [conv_l(node_feats[:, :, idx]) for idx, conv_l in enumerate(self.conv_layers)]
        node_feats = torch.cat(node_feats, dim=-1)
        graph.node.update_feats(self.relu_l(node_feats))
        return graph


class EdgeFeatLayer(nn.Module):
    """
    Calculate the features of edges. The inputs should contain the features of nodes, condition variables and previous
    edges, however, now we only consider setting the features of nodes, condition variables as inputs.
    """
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 param: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.param = param
        if param in ('share', 'none'):
            self.feat_l = None
        else:
            n2n_method, n2c_method = param.split('^')
            if n2c_method == 'film':
                self.feat_l = EdgeFeatFilm(node_dim, cond_dim, edge_dim, n2n_method, dropout)
            elif n2c_method == 'mul':
                self.feat_l = EdgeFeatMul(node_dim, cond_dim, edge_dim, n2n_method, dropout)
            elif n2c_method == 'cat':
                self.feat_l = EdgeFeatCat(node_dim, cond_dim, edge_dim, n2n_method, dropout)
            else:
                raise NotImplementedError()
        self.node_dim = node_dim
        self.cond_dim = cond_dim
        self.edge_dim = edge_dim

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}_{self.edge_dim}'

    def forward(self, graph: Graph):
        if self.param == 'none':
            return graph
        feat_l = self.feat_l or graph.edge.feat_layers[self.layer_key]
        if self.layer_key not in graph.edge.feat_layers:
            graph.edge.feat_layers[self.layer_key] = self.feat_l
        edge_feats = feat_l(graph)  # m, c, has been processed by remove_op
        graph.edge.edge_attrs['feats'] = EdgeAttr('feats', edge_feats, graph.edge.init_op())
        return graph


class EdgeWeightLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        if '_' in method:
            self.method, method = str_split(method, '_')
        else:
            self.method = 'cond'
        self.method_param, self.norm_method, self.reduce_size = str_split(method, '^')
        if self.method == 'cond':
            if self.method_param == 'share':
                self.logit_l = None
            else:
                self.logit_l = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.utils.weight_norm(nn.Linear(edge_dim, edge_dim//2)),
                    nn.ReLU(),
                    nn.utils.weight_norm(nn.Linear(edge_dim//2, int(kernel_size)))
                )
        elif self.method == 'cgs':
            if self.method_param == 'share':
                self.logit_l = None
            else:
                self.logit_l = nn.Sequential(
                    nn.utils.weight_norm(nn.Linear(node_dim + cond_dim, edge_dim)),
                    nn.ReLU(),
                    nn.utils.weight_norm(nn.Linear(edge_dim, edge_dim)),
                    nn.ReLU()
                )
        else:
            raise NotImplementedError()
        self.drop_l = nn.Dropout(dropout)

    @property
    def layer_key(self):
        return f'{self.edge_dim}'

    def forward(self, graph: Graph):
        logit_l = self.logit_l or graph.edge.logit_layers[self.layer_key]
        if self.layer_key not in graph.edge.logit_layers:
            graph.edge.logit_layers[self.layer_key] = self.logit_l
        if self.method == 'cond':
            edge_feats = graph.edge.edge_attrs['feats']
            edge_logits = EdgeAttr('logits', logit_l(edge_feats.value), edge_feats.op)

            mirror_op = EdgeMirror(edge_feats.op, 'mirror')
            edge_logits = mirror_op.attr_process(edge_logits)

            graph.edge.edge_attrs['logits'] = edge_logits
            edge_weights = mirror_op.norm(edge_logits.value, self.norm_method)

            topk_op = EdgeTopK(edge_weights, self.reduce_size, edge_logits.op, 'topk', keep_self=True)
            topk_weights = topk_op.by_attr.view(-1, topk_op.by_attr.shape[-1])
            graph.edge.edge_attrs['weights'] = EdgeAttr('weights', topk_weights, topk_op)
        elif self.method == 'cgs':
            if self.node_dim % 512 == 4:
                coord_feats = torch.cat(graph.node.spatial_attr, dim=-1)
                node_feats = self.drop_l(graph.node_feats)
                node_feats = torch.cat((node_feats, coord_feats), dim=-1)
            else:
                node_feats = self.drop_l(graph.node_feats)
            node_q_feats = torch.cat((node_feats, graph.cond_feats.unsqueeze(dim=1).expand(-1, graph.node_num, -1)), dim=-1)
            joint_f = logit_l(node_q_feats)
            edge_logits = torch.matmul(joint_f, joint_f.transpose(1, 2)).view(-1, 1)
            edge_logits = EdgeAttr('logits', edge_logits, graph.edge.init_op())
            mirror_op = EdgeMirror(edge_logits.op, 'mirror')
            edge_logits = mirror_op.attr_process(edge_logits)

            graph.edge.edge_attrs['logits'] = edge_logits
            edge_weights = mirror_op.norm(edge_logits.value, self.norm_method)

            topk_op = EdgeTopK(edge_weights, self.reduce_size, edge_logits.op, 'topk')
            topk_weights = topk_op.by_attr.view(-1, topk_op.by_attr.shape[-1])
            graph.edge.edge_attrs['weights'] = EdgeAttr('weights', topk_weights, topk_op)
        else:
            raise NotImplementedError()
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
        self.method = method
        if method in ('share', 'none'):
            self.param_l = None
        else:
            self.param_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(edge_dim, out_dim//2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(out_dim//2, out_dim)),
                nn.Tanh()
            )

    @property
    def layer_key(self):
        return f'{self.edge_dim}_{self.out_dim}'

    def forward(self, graph: Graph):
        if self.method == 'none':
            return graph
        param_l = self.param_l or graph.edge.param_layers[self.layer_key]
        if self.layer_key not in graph.edge.param_layers:
            graph.edge.param_layers[self.layer_key] = self.param_l

        edge_feats = graph.edge_attrs['feats']
        edge_params = EdgeAttr('params', param_l(edge_feats.value), edge_feats.op)  # has been processed by remove op

        edge_weights = graph.edge_attrs['weights']
        edge_params = edge_weights.op.attr_process(edge_params)
        graph.edge.edge_attrs['params'] = edge_params
        return graph


class NodeFeatLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 out_dim: int,
                 method: str,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.method, self.method_param = str_split(method, '_')
        if self.method == 'cond':
            self.node_feat_l = CondNodeFeat(node_dim, cond_dim, out_dim, self.method_param, dropout)
        elif self.method == 'cgs':
            self.node_feat_l = nn.Sequential(
                CgsNodeFeat(node_dim, out_dim, self.method_param, dropout=dropout),
                CgsNodeFeat(out_dim, out_dim, self.method_param, use_graph_weights=False, dropout=dropout)
            )

    @property
    def layer_key(self):
        return f'{self.node_dim}_{self.cond_dim}'

    def forward(self, graph: Graph):
        graph = self.node_feat_l(graph)
        return graph


class NodeWeightLayer(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 method: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.method = method
        if method == 'none':
            self.node_logit_l = None
            return
        self.weight_method, self.node_method, self.norm_method = str_split(method, '^')
        if self.node_method == 'linear':
            self.node_logit_l = nn.Sequential(
                nn.Dropout(dropout),
                nn.utils.weight_norm(nn.Linear(node_dim, node_dim // 2)),
                nn.ReLU(),
                nn.utils.weight_norm(nn.Linear(node_dim // 2, 1))
            )
        else:
            self.node_logit_l = None

    @property
    def layer_key(self):
        return f'{self.node_dim}'

    def norm(self, logits):
        if self.norm_method == 'sigmoid':
            return logits.sigmoid()
        elif self.norm_method == 'softmax':
            return logits.softmax(dim=1)
        else:
            raise NotImplementedError()

    def forward(self, graph: Graph):
        if self.method == 'none':
            return graph
        if self.node_method == 'linear':
            node_logits = self.node_logit_l(graph.node_feats)
            graph.node.logit_layers[self.layer_key] = self.node_logit_l
        elif self.node_method == 'share':
            node_logits = graph.node.logit_layers[self.layer_key](graph.node_feats)
        elif self.node_method == 'inherit':
            node_weights = graph.node.weights
        else:
            raise NotImplementedError()

        if self.node_method in ('linear', 'share'):
            b_num, node_num = graph.batch_num, graph.node_num
            if self.weight_method == 'all':
                edge_logits = graph.edge.edge_attrs['logits']
                edge_weights = edge_logits.op.norm(edge_logits.value, 'softmax').view(b_num, node_num, -1, 1)
                node_j_logits = node_logits.view(-1, 1)[edge_logits.op.coords[1]].view(b_num, node_num, -1, 1)
                nb_logits = edge_weights * node_j_logits
                nb_logits = nb_logits.sum(dim=2)
                node_logits = nb_logits
            node_weights = self.norm(node_logits)

        # batch_num, node_num = graph.batch_num, graph.node_num
        # if self.weight_method == 'all':
        #     edge_logits = graph.edge.edge_attrs['logits']
        #     edge_weights = edge_logits.op.norm(edge_logits.value, 'softmax').view(batch_num, node_num, -1, 1)
        #     node_j_weights = node_weights.view(-1, 1)[edge_logits.op.coords[1]].view(batch_num, node_num, -1, 1)
        #     nb_weights = edge_weights * node_j_weights
        #     nb_weights = nb_weights.sum(dim=2)
        #
        #     if not edge_logits.op.is_loop:
        #         node_weights = node_weights * 0.5 + nb_weights * 0.5
        graph.node.weights = node_weights
        return graph


class GraphLearner(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 param,
                 dropout: float = 0.
                 ):
        super().__init__()
        feat_param, weight_param = param['feat'], param['weight']
        self.edge_feat_l = EdgeFeatLayer(node_dim, cond_dim, edge_dim, feat_param, dropout)
        self.edge_weight_l = EdgeWeightLayer(node_dim, cond_dim, edge_dim, weight_param, dropout)

    def forward(self, graph):
        graph = self.edge_feat_l(graph)
        graph = self.edge_weight_l(graph)
        return graph


class GraphConv(nn.Module):
    def __init__(self,
                 node_dim: int,
                 cond_dim: int,
                 edge_dim: int,
                 out_dim: int,
                 params: str,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.params = params
        feat_param, param_param, node_param, weight_param = (params[key] for key in ('feat', 'param', 'node', 'weight'))
        self.edge_feat_l = EdgeFeatLayer(node_dim, cond_dim, edge_dim, feat_param, dropout)
        self.edge_params_l = EdgeParamLayer(edge_dim, out_dim, param_param, dropout)
        self.node_feat_l = NodeFeatLayer(node_dim, cond_dim, out_dim, node_param, dropout)
        self.node_weight_l = NodeWeightLayer(out_dim, edge_dim, weight_param, dropout)

    def forward(self, graph):
        graph = self.edge_feat_l(graph)
        graph = self.edge_params_l(graph)
        graph = self.node_feat_l(graph)
        graph = self.node_weight_l(graph)
        return graph


class GraphPool(nn.Module):
    def __init__(self,
                 params: str
                 ):

        super().__init__()
        self.params = params
        self.pool_params, self.reduce_params = params.split('_')

    def forward(self, graph):
        if self.pool_params != 'none':
            graph.pool_feats(self.pool_params)
        if self.reduce_params == 'none':
            return graph
        reduce_size = int(self.reduce_params)
        node_weights = graph.node_weights
        node_weights, top_idx = node_weights.topk(reduce_size, dim=1, sorted=False)  # b,max_size,1
        c_num = graph.node_feats.size(-1)
        graph.node.feats = graph.node_feats.gather(index=top_idx.expand(-1, -1, c_num), dim=1)
        graph.node.boxes = graph.node.boxes.gather(index=top_idx.expand(-1, -1, 4), dim=1)
        graph.node.weights = node_weights
        if graph.node.mask is not None:
            new_mask = graph.node_mask.gather(index=top_idx, dim=1)
            if (new_mask == False).sum() == 0:
                graph.node._mask = None
                graph.node.mask_nums = None
            else:
                graph.node._mask = new_mask
        # graph.node.feats = graph.node_weights * graph.node_feats

        new_edge = Edge(graph.node, graph.edge.method)
        new_edge.feat_layers = graph.edge.feat_layers
        new_edge.logit_layers = graph.edge.logit_layers
        new_edge.param_layers = graph.edge.param_layers
        graph.edge = new_edge
        return graph






































































