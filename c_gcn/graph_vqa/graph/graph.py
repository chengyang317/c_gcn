# coding=utf-8
import torch
from .node import Node
from .edge import Edge
import numpy as np


__all__ = ['Graph']


class Graph(object):
    def __init__(self,
                 node_feats,
                 node_coords=None,
                 node_weights=None,
                 cond_feats=None,
                 filter_method: str = 'tri_u',
                 ):
        self.node = Node(self, node_feats, node_coords, node_weights)
        self.edge = Edge(self.node, filter_method)
        self.cond_feats = cond_feats
        self.feats = list()

    @property
    def device(self):
        return self.node.device

    @property
    def node_num(self):
        return self.node.node_num

    @property
    def batch_num(self):
        return self.node.batch_num

    @property
    def edge_num(self):
        return self.edge.edge_num

    @property
    def node_feats(self):
        return self.node.feats

    @property
    def edge_feats(self):
        return self.edge.feats

    @property
    def edge_attrs(self):
        return self.edge.edge_attrs

    @property
    def node_weights(self):
        return self.node.weights

    @property
    def edge_weights(self):
        return self.edge.weights

    def pooling_feats(self, method='mean'):
        if 'weight' in method:
            method = method.split('^')[-1]
            node_feats = self.node_feats * self.node_weights
        else:
            node_feats = self.node_feats

        if method == 'mean':
            return node_feats.mean(dim=1).squeeze()
        elif method == 'max':
            return node_feats.max(dim=1)[0]
        elif method == 'sum':
            return node_feats.sum(dim=1)
        elif method == 'mix':
            max_feat = node_feats.max(dim=1)[0]
            mean_feat = node_feats.mean(dim=1).squeeze()
            return torch.cat((max_feat, mean_feat), dim=-1)
        else:
            raise NotImplementedError()
        # if self.node_weights is not None:
        #     if self.node_weights.dim() == 2:
        #         self.node.weights = self.node_weights[:, :, None]
        #     node_feats = self.node_weights.mean(dim=-1, keepdim=True) * self.node_feats
        # else:
        #     node_feats = self.node_feats
        # return node_feats.mean(dim=1)

    def graph_feats(self, method):
        if method == 'last':
            return self.feats[-1]
        elif method == 'cat':
            return torch.cat(self.feats, dim=-1)
        elif method == 'sum':
            return sum(self.feats)
        elif method == 'max':
            return torch.stack(self.feats, dim=1).max(dim=1)[0]
        elif method == 'mean':
            return torch.stack(self.feats, dim=1).mean(dim=1)
        else:
            raise NotImplementedError()

    def calc_node_weights(self):
        # node_weights = self.node.weights or torch.tensor([1. / self.node_num]).cuda(self.device)[None, :]\
        #     .expand(self.batch_num, self.node_num).unsqueeze(-1)  # b, n, 1
        if self.node.weights is None:
            node_weights = torch.randn(self.node_num).div(10).add(1.).cuda(self.device)[None, :].\
                expand(self.batch_num, self.node_num).unsqueeze(-1).contiguous()
        else:
            node_weights = self.node_weights
        edge_logits = self.edge.edge_attrs['logits']

        edge_weights = edge_logits.op.norm(edge_logits.value, 'softmax').view(self.batch_num, self.node_num, -1, 1)
        node_j_weights = node_weights.view(-1, 1)[edge_logits.op.coords[1]].view(self.batch_num, self.node_num, -1, 1)
        nb_weights = edge_weights * node_j_weights
        nb_weights = nb_weights.sum(dim=2)

        if not edge_logits.op.is_loop:
            # node_weights += nb_weights  # b, n, 1
            node_weights = node_weights * 0.5 + nb_weights * 0.5
        self.node.weights = node_weights
        return node_weights






































