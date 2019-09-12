import torch
import collections
from .edge import EdgeTopK


__all__ = ['Node']


class Node(object):
    def __init__(self, graph, node_feats, node_boxes=None, node_weights=None, node_valid_nums=None):
        assert node_feats.dim() == 3
        self.graph = graph
        self.batch_num, self.node_num, self.feat_num = node_feats.shape
        self.boxes = node_boxes
        self.weights = node_weights
        self.valid_nums = node_valid_nums
        self.feat_layers = collections.defaultdict(None)
        self.logit_layers = {}
        self.mask = None
        self.old2new_map = torch
        if self.valid_nums is not None:
            self.mask = torch.arange(self.node_num).expand(self.batch_num, -1).cuda(self.device) < self.valid_nums[:, None]
            self.feats = node_feats[self.mask]
            self.idx_map
        else:
            self.mask = torch.ones(self.batch_num, self.node_num).cuda(self.device).bool()
            self.feats = node_feats.view(-1, self.feat_num)

    @property
    def edge(self):
        return self.graph.edge

    @property
    def indexes(self):
        return torch.arange(self.node_num*self.batch_num, device=self.device)

    @property
    def device(self):
        return self.feats.device

    def update_feats(self, node_feats=None, node_boxes=None, node_weights=None):
        if node_feats is not None:
            self.feats = node_feats
        if node_boxes is not None:
            self.boxes = node_boxes
        if node_weights is not None:
            self.weights = node_weights

    @property
    def size_center(self):
        boxes = self.boxes
        node_size = (boxes[:, :, 2:] - boxes[:, :, :2])
        node_centre = boxes[:, :, :2] + 0.5 * node_size  # b, k, 2
        return node_size, node_centre

