import torch
import collections
from typing import Dict
from pt_pack import node_intersect
import torch_scatter as ts


__all__ = ['Edge', 'EdgeAttr', 'EdgeNull', 'EdgeTopK',]


EdgeAttr = collections.namedtuple('EdgeAttr', ['name', 'value', 'op'])


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    exps = exps * mask.float()
    masked_sums = exps.sum(dim, keepdim=True) + epsilon
    return exps / masked_sums


class EdgeOp(object):
    op_name = 'BASE'
    caches = {}

    def __init__(self, name, last_op):
        self.name = name or self.op_name
        self.last_op = last_op
        self.node = None
        if last_op is not None:
            last_op.register_op(self)
            self.node = last_op.node
        self.edge_attrs: Dict[str, EdgeAttr] = {}
        self.node_i_ids, self.node_j_ids = None, None
        self.next_ops = {}
        self.mask = None
        self.select_ids = None
        self.op_process()

    @property
    def edge_num(self):
        return self.node_j_ids.shape[0]

    @property
    def eye_mask_cache(self):
        key = f'eye_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            eye_mask = torch.eye(self.node_num).expand(self.batch_num, -1, -1).contiguous().bool()
            self.caches[key] = eye_mask
        return self.caches[key]

    @property
    def meshgrid_cache(self):
        key = f'meshgrid_{self.batch_num}_{self.node_num}'
        if key not in self.caches:
            batch_idx, node_i, node_j = torch.meshgrid(torch.arange(self.batch_num) * self.node_num,
                                                       torch.arange(self.node_num), torch.arange(self.node_num)
                                                       )
            node_i = batch_idx + node_i
            node_j = batch_idx + node_j
            self.caches[key] = (node_i, node_j)
        return self.caches[key]

    def op_process(self):
        raise NotImplementedError()

    def _attr_process(self, attr: torch.Tensor):
        raise NotImplementedError

    def attr_process(self, attr):
        if attr is None:
            return None
        if isinstance(attr, EdgeAttr):
            if attr.op.name == self.last_op.name:
                attr_value = self._attr_process(attr.value)
            elif attr.op.name == self.name:
                return attr
            else:
                attr_value = self._attr_process(self.last_op.attr_process(attr).value)
            return EdgeAttr(attr.name, attr_value, self)
        return self._attr_process(attr)

    @property
    def node_num(self):
        return self.node.node_num

    @property
    def batch_num(self):
        return self.node.batch_num

    @property
    def device(self):
        return self.node.device

    def register_op(self, op):
        self.next_ops[op.name] = op

    def load_op(self, op_name):
        return self.next_ops.get(op_name, None)

    def clear_ops(self):
        self.next_ops = {}

    def norm(self, edge_attr, method):
        if method == 'softmax':
            exp = edge_attr.exp()
            sums = ts.scatter_add(exp, self.node_i_ids)
            return exp / sums[self.node_i_ids]
        else:
            raise NotImplementedError()

    def reshape(self, edge_attr, fill_value=0.):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        c_dim = edge_attr.size(-1)
        new_attr = edge_attr.new_full((self.batch_num, self.node_num, self.node_num, c_dim), fill_value)
        new_attr[self.mask] = edge_attr
        return new_attr


class EdgeNull(EdgeOp):
    op_name = 'NULL'

    def __init__(self, node):
        super().__init__('null', None)
        self.node = node

    def attr_process(self, attr):
        if attr is None:
            return None
        if isinstance(attr, EdgeAttr):
            assert isinstance(attr.op, EdgeNull)
        return attr

    def op_process(self):
        pass


class EdgeInit(EdgeOp):
    op_name = 'INIT'

    def __init__(self, node):
        super().__init__(f'init', EdgeNull(node))

    def op_process(self):
        edge_mask = node_intersect(self.node.mask.unsqueeze(-1), 'mul')  # b, n, n
        node_i, node_j = self.meshgrid_cache  # b, n, n
        node_i, node_j = node_i.cuda(self.device)[edge_mask], node_j.cuda(self.device)[edge_mask]  # k, k
        self.node_i_ids, self.node_j_ids = self.node.old2new_map[node_i], self.node.old2new_map[node_j]
        self.mask = edge_mask

    def _attr_process(self, attr: torch.Tensor):
        attr = attr.view(-1, attr.shape[-1])
        assert attr.shape[0] == self.node_num * self.node_num * self.batch_num
        return attr[self.indexes]


class EdgeTopK(EdgeOp):
    op_name = 'TOPK'

    def __init__(self, by_attr: torch.Tensor, reduce_size, last_op, name=None, keep_self=False):
        self.by_attr = by_attr
        self.reduce_size = min(self.node.min_node_num, reduce_size)
        self.top_ids = None
        self.keep_self = keep_self
        name = name or f'top_{reduce_size}'
        super().__init__(name, last_op)

    def op_process(self):
        last_op = self.last_op
        by_attr = self.reshape(self.by_attr, fill_value=-100.)
        by_attr, top_ids = self.attr_topk(by_attr, -2, self.reduce_size, keep_self=self.keep_self)  # b, n_num, k, 1
        self.top_ids = top_ids.squeeze(-1)  # b, n_num, k
        select_ids = self.reshape(torch.arange(last_op.edge_num).cuda(self.device))
        self.select_ids = select_ids.gather(index=)

    def attr_topk(self, attr, dim, reduce_size, keep_self=False):
        """

        :param attr: b_num, n_num, -1, k_num or b_num, n_num, k_num
        :param dim:
        :param reduce_size:
        :param use_abs:
        :return: o_b, n_num, k,
        """
        k_size = attr.shape[-1]
        if k_size > 1:
            attr = attr.mean(dim=-1, keepdim=True)  # o_b,**, 1

        if not keep_self:
            attr, top_indexes = attr.topk(reduce_size, dim=dim, sorted=False)  # b,obj_num,max_size,k_size
        else:
            loop_mask = self.eye_mask_cache.cuda(self.device)
            fake_attr = attr.masked_fill(loop_mask, 1000.)
            _, top_indexes = fake_attr.topk(reduce_size, dim=dim, sorted=False)
            attr = attr.gather(index=top_indexes, dim=-2)
        return attr, top_indexes

    def _attr_process(self, attr: torch.Tensor):
        attr = self.reshape(attr, dim_is_4=True)
        assert attr.shape[0] * attr.shape[1] * attr.shape[2] == self.last_op.edge_num
        attr_c = attr.shape[-1]
        return attr.gather(index=self.top_ids.unsqueeze(-1).expand(-1, -1, -1, attr_c), dim=2).view(-1, attr_c)


class Edge(EdgeInit):
    op_name = 'EDGE_INIT'

    def __init__(self,
                 node,
                 init_method: str,
                 ):
        self.graph = node.graph
        self.edge_attrs = {
            'feats': None, 'params': None, 'weights': None
        }
        self.feat_layers, self.logit_layers, self.param_layers = {}, {}, {}
        super().__init__(init_method, node)

    @property
    def node(self):
        return self.graph.node

    @property
    def feat_dim(self):
        return self.feats.shape[-1]

    def init_op(self):
        return self

    def remove_loop_op(self) -> EdgeOp:
        if 'remove_loop' not in self.next_ops:
            EdgeRemoveLoop(self, 'remove_loop')
        return self.next_ops['remove_loop']

    def mirror_op(self) -> EdgeOp:
        if 'mirror' not in self.next_ops:
            EdgeMirror(self, 'mirror')
        return self.next_ops['mirror']

    def topk_op(self, by_attr=None, reduce_size=None) -> EdgeTopK:
        mirror_op = self.mirror_op()
        if reduce_size is None:
            for op_name, op in mirror_op.next_ops.items():
                if 'top' in op_name:
                    return op
            raise NotImplementedError()
        if f'top_{reduce_size}' not in mirror_op.next_ops:
            EdgeTopK(by_attr, reduce_size, mirror_op, f'top_{reduce_size}')
        return mirror_op.next_ops[f'top_{reduce_size}']

    def update(self, edge_indexes=None, edge_coords=None, edge_weights=None, dir_flag=None, loop_flag=None):
        assert (edge_indexes is not None) + (edge_coords is not None) != 2
        if edge_indexes is not None:
            self._indexes = edge_indexes
            self._coords = None
        if edge_coords is not None:
            self._coords = edge_coords
            self._indexes = None
        if edge_weights is not None:
            self.weights = edge_weights
        if dir_flag is not None:
            self._dir_flag = dir_flag
        if loop_flag is not None:
            self._loop_flag = None

    def update_by_index(self, top_indexes):
        top_indexes = self.reshape(top_indexes)
        self._indexes = self.reshape(self.indexes).gather(index=top_indexes, dim=2).view(-1)
        self._coords = None
        if self.feats is not None:
            self.feats = self.reshape(self.feats, True).gather(index=top_indexes, dim=2).view(-1, self.feat_dim)

    def aggregate(self, node_feats, edge_weights: EdgeAttr, method='sum'):
        add_op = EdgeAddLoop(edge_weights.op)
        edge_weights = add_op.attr_process(edge_weights).value
        b_num, n_num, c_num = node_feats.shape
        node_j_feats = node_feats.view(b_num * n_num, c_num)[add_op.coords[1]]  # m, o_c
        _, k_size = edge_weights.shape
        if k_size == c_num:
            weighted_feats = edge_weights * node_j_feats  # m, c_num
        else:
            weighted_feats = edge_weights[:, :, None] * node_j_feats[:, None, :]  # m, k_size, o_c
        if method == 'sum':
            node_k_feats = ts.scatter_add(weighted_feats, add_op.coords[0], dim=0)  # obj_num*b,k_size,o_c
        elif method == 'max':
            node_k_feats = ts.scatter_max(weighted_feats, add_op.coords[0], dim=0)[0]
        elif method == 'mean':
            node_k_feats = ts.scatter_mean(weighted_feats, add_op.coords[0], dim=0)
        else:
            raise NotImplementedError()
        node_feats = node_k_feats.view(b_num, n_num, *weighted_feats.shape[1:])  # b, obj_num, out_dim or b, o_n, k_n, c
        return node_feats.squeeze()

    def edge2node(self, attr: EdgeAttr, method='sum'):
        value = self.reshape(attr.value, dim_is_4=True)
        if method == 'sum':
            return value.sum(dim=2)
        else:
            raise NotImplementedError()

    def load_node_attr(self, node_attr):
        return node_attr[self.node_i_ids], node_attr[self.node_j_ids]

    def load_spatial_feats(self, node_boxes):
        node_size = (node_boxes[:, 2:] - node_boxes[:, :2])
        # node_centre = node_boxes[:, :2] + 0.5 * node_size  # b*k, 2
        node_i_box, node_j_box = self.load_node_attr(node_boxes)
        node_i_size, node_j_size = self.load_node_attr(node_size)

        node_dist = node_i_box - node_j_box
        node_dist = node_dist / node_j_size
        node_scale = node_i_size / node_j_size
        node_mul = (node_i_size[:, 0] * node_j_size[:, 1]) / (node_j_size[:, 0] * node_j_size[:, 1])
        node_sum = (node_i_size[:, 0] * node_j_size[:, 1]) / (node_j_size[:, 0] + node_j_size[:, 1])
        return torch.cat((node_dist, node_scale, node_mul, node_sum), dim=-1)

    def load_feats(self, node_feats, n2n_method):
        node_i_feats, node_j_feats = node_feats[self.node_i_ids], node_feats[self.node_j_ids]
        if n2n_method == 'cat':
            joint_feats = node_i_feats + node_j_feats
        elif n2n_method == 'mul':
            joint_feats = node_i_feats * node_j_feats
        else:
            raise NotImplementedError()
        return joint_feats









