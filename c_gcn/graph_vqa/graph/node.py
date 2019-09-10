import torch
import collections
from .edge import EdgeTopK


__all__ = ['Node']


class Node(object):
    def __init__(self, graph, node_feats, node_coords=None, node_weights=None):
        assert node_feats.dim() == 3
        self.graph = graph
        self.feats = node_feats
        self.coords = node_coords
        self.weights = node_weights
        self.feat_layers = collections.defaultdict(None)
        self.logit_layers = {}
        self.logits = None

    @property
    def edge(self):
        return self.graph.edge

    @property
    def indexes(self):
        return torch.arange(self.node_num*self.batch_num, device=self.device)

    @property
    def node_num(self):
        return self.feats.shape[1]

    @property
    def batch_num(self):
        return self.feats.shape[0]

    @property
    def feat_dim(self):
        return self.feats.shape[-1]

    @property
    def device(self):
        return self.feats.device

    def update_feats(self, node_feats=None, node_coords=None, node_weights=None):
        if node_feats is not None:
            self.feats = node_feats
        if node_coords is not None:
            self.coords = node_coords
        if node_weights is not None:
            self.weights = node_weights

    @property
    def shape(self):
        return self.feats.shape

    @property
    def spatial_attr(self):
        node_coord = self.coords
        node_size = (node_coord[:, :, 2:] - node_coord[:, :, :2])
        node_centre = node_coord[:, :, :2] + 0.5 * node_size  # b, k, 2
        return node_size, node_centre

    def pool(self, logits, norm_method, reduce_size=-1):
        assert logits.shape[:2] == (self.batch_num, self.node_num)
        if logits.shape[-1] != 1:
            logits = logits.mean(dim=-1, keepdim=True)
        logits, top_idx = EdgeTopK.attr_topk(logits, dim=1, reduce_size=reduce_size)
        self.weights = self.norm(logits, norm_method)  # b, node_num, 1
        if top_idx is not None:
            self.feats = self.feats.gather(index=top_idx.expand(-1, -1, self.feats.shape[-1]), dim=1)
            self.coords = self.coords.gather(index=top_idx.expand(-1, -1, self.coords.shape[-1]), dim=1)
        self.feats = (self.weights + 1) * self.feats
        return self

    def logit2weight(self, norm_method, reduce_size=-1):
        edge_logits = self.edge.logits
        node_logits = self.edge.edge2node(edge_logits)  # b,n_num, k
        node_logits, top_indexes = logit_reduce(node_logits, 1, reduce_size)
        node_weights = logit_norm(node_logits, norm_method)
        self.weights = node_weights.squeeze()  # b, n_num
        if top_indexes is not None:
            self.feats = self.feats.gather(index=top_indexes.expand(-1, -1, self.feats.shape[-1]), dim=1)
            self.coords = self.coords.gather(index=top_indexes.expand(-1, -1, self.coords.shape[-1]), dim=1)
        return node_weights

    def norm(self, attr, method):
        if method == 'softmax':
            ret_attr = attr.softmax(dim=-2)  # b, obj_num, n, k_size
            # weight = weight + 1
        elif method == 'tanh':
            ret_attr = attr.tanh()
            # weight = weight + 1  # major change
            # node_weight = node_logit.tanh()
        elif method == 'sigmoid':
            ret_attr = attr.sigmoid()
            # weight = weight + 1  # major change
        elif method == 'self':
            ret_attr = attr
        else:
            raise NotImplementedError()
        return ret_attr