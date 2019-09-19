# coding=utf-8
import torch
from .node import Node
from .edge import Edge
import torch_scatter as ts


__all__ = ['Graph']


class Graph(object):
    def __init__(self,
                 node_feats,
                 node_coords=None,
                 node_weights=None,
                 cond_feats=None,
                 q_ids=None,
                 ):
        self.node = Node(self, node_feats, node_coords, node_weights)
        self.edge = Edge(self.node)
        self.cond_feats = cond_feats
        self.layer_feats = list()

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

    def pool_feats(self, method='mean'):
        if 'weight' in method:
            method = method.split('^')[-1]
            node_feats = self.node_feats * self.node_weights
        else:
            node_feats = self.node_feats

        if method == 'mean':
            feats = ts.scatter_mean(node_feats, self.node.batch_ids, dim=0)
        elif method == 'max':
            feats = ts.scatter_max(node_feats, self.node.batch_ids, dim=0)[0]
        elif method == 'sum':
            feats = ts.scatter_sum(node_feats, self.node.batch_ids, dim=0)
        elif method == 'mix':
            max_feat = ts.scatter_max(node_feats, self.node.batch_ids, dim=0)[0]
            mean_feat = ts.scatter_mean(node_feats, self.node.batch_ids, dim=0)
            feats = torch.cat((max_feat, mean_feat), dim=-1)
        else:
            raise NotImplementedError()
        self.layer_feats.append(feats)
        return feats

    def graph_feats(self, method):
        if method == 'last':
            return self.layer_feats[-1]
        elif method == 'cat':
            return torch.cat(self.layer_feats, dim=-1)
        elif method == 'sum':
            return sum(self.layer_feats)
        elif method == 'max':
            return torch.stack(self.layer_feats, dim=1).max(dim=1)[0]
        elif method == 'mean':
            return torch.stack(self.layer_feats, dim=1).mean(dim=1)
        else:
            raise NotImplementedError()








































