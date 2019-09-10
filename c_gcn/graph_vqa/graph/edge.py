import torch
import collections
from typing import Dict
from pt_pack import node_intersect
# import torch_scatter as ts


__all__ = ['Edge', 'EdgeAttr', 'EdgeNull', 'EdgeAddLoop', 'EdgeRemoveLoop', 'EdgeTopK', 'EdgeMirror']


EdgeAttr = collections.namedtuple('EdgeAttr', ['name', 'value', 'op'])


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    exps = exps * mask.float()
    masked_sums = exps.sum(dim, keepdim=True) + epsilon
    return exps / masked_sums


class EdgeOp(object):
    op_name = 'BASE'

    def __init__(self, name, last_op):
        self.name = name or self.op_name
        self.last_op = last_op
        if last_op is not None:
            last_op.register_op(self)
        self._indexes, self._coords, self._mirror_flag, self._loop_flag = (None,) * 4
        self.edge_attrs: Dict[str, EdgeAttr] = {}
        self.next_ops = {}
        self.op_process()

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
    def indexes(self):
        """

        :return: m,
        """
        if self._indexes is None:
            self._indexes = self.coord2ids(self._coords, self.node_num)
        return self._indexes

    @indexes.setter
    def indexes(self, edge_indexes):
        self._indexes = edge_indexes
        self._coords = None

    @property
    def coords(self):
        if self._coords is None:
            self._coords = self.ids2coord(self._indexes, self.node_num)
        return self._coords

    @coords.setter
    def coords(self, edge_coords):
        self._coords = edge_coords
        self._indexes = None

    @property
    def node_num(self):
        if hasattr(self, '_node_num'):
            return self._node_num
        return self.last_op.node_num

    @property
    def batch_num(self):
        if hasattr(self, '_batch_num'):
            return self._batch_num
        return self.last_op.batch_num

    def register_op(self, op):
        self.next_ops[op.name] = op

    def load_op(self, op_name):
        return self.next_ops.get(op_name, None)

    @classmethod
    def ids2coord(cls, edge_ids, obj_num):
        """

        :param edge_ids: m
        :param obj_num:
        :return:
        """
        b_ids = edge_ids // (obj_num * obj_num)
        ids = edge_ids % (obj_num * obj_num)
        b_row = ids // obj_num
        b_col = ids % obj_num
        edge_row = b_row + b_ids * obj_num
        edge_col = b_col + b_ids * obj_num
        return torch.stack((edge_row, edge_col), dim=0)

    @classmethod
    def coord2ids(cls, edge_coord, obj_num):
        b_ids = edge_coord[0] // obj_num
        b_row, b_col = edge_coord % obj_num
        ids = b_row * obj_num + b_col
        edge_ids = ids + b_ids * obj_num * obj_num
        return edge_ids

    @property
    def is_mirror(self):
        if self._mirror_flag is None:
            self._mirror_flag = self.check_mirror(self.coords, self.indexes, self.node_num)
        return self._mirror_flag

    @property
    def is_loop(self):
        if self._loop_flag is None:
            self._loop_flag = self.check_loop(self.coords)
        return self._loop_flag

    @classmethod
    def check_mirror(cls, edge_coord, edge_ids, node_num):
        edge_row, edge_col = edge_coord
        new_edge_ids = cls.coord2ids(torch.stack((edge_col, edge_row), dim=0), node_num)
        return set(edge_ids.tolist()) == set(new_edge_ids.tolist())

    @classmethod
    def check_loop(cls, edge_coord):
        row, col = edge_coord
        mask = row == col
        return mask.sum().item() > 0

    @property
    def edge_num(self):
        return len(self.indexes)

    def reshape(self, x: torch.Tensor, dim_is_4=False):
        if dim_is_4:
            return x.view(self.batch_num, self.node_num, -1, x.shape[-1])
        return x.view(self.batch_num, self.node_num, -1)

    def clear_ops(self):
        self.next_ops = {}

    def loop_mask(self):
        row, col = self.coords
        mask = row == col
        return mask

    def norm(self, attr, method):
        attr = self.reshape(attr, dim_is_4=True)
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
        return ret_attr.view(-1, ret_attr.shape[-1])


class EdgeNull(EdgeOp):
    op_name = 'NULL'

    def __init__(self):
        super().__init__('null', None)

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
    ids_cache = {}

    def __init__(self, method: str, batch_num, node_num, device=None):
        self._batch_num, self._node_num = batch_num, node_num
        self.method = method
        self.device = device
        super().__init__(f'init_{method}', EdgeNull())

    def op_process(self):
        key = f'{self.batch_num}_{self.node_num}_{self.method}'
        if key not in self.ids_cache:
            self.ids_cache[key] = self.batch_filter_ids(self.batch_num, self.node_num, self.method, self.device)
        self.indexes = self.ids_cache[key]
        self._mirror_flag, self._loop_flag = self.check_flag(self.method)

    def _attr_process(self, attr: torch.Tensor):
        attr = attr.view(-1, attr.shape[-1])
        assert attr.shape[0] == self.node_num * self.node_num * self.batch_num
        return attr[self.indexes]

    @classmethod
    def batch_filter_coord(cls, b_num, obj_num, filter_method, device=None):
        edge_ids = cls.batch_filter_ids(b_num, obj_num, filter_method, device)
        return cls.ids2coord(edge_ids, obj_num)

    @classmethod
    def batch_filter_ids(cls, b_num, obj_num, filter_method, device=None):
        edge_ids = cls.filter_ids(obj_num, filter_method, device)
        b_prefix = torch.arange(b_num) * obj_num * obj_num
        edge_ids = edge_ids[None] + b_prefix.cuda(device)[:, None]
        return edge_ids.view(-1)

    @classmethod
    def filter_ids(cls, obj_num, filter_method, device=None):
        if filter_method == 'full':
            filter_ids = torch.arange(0, obj_num * obj_num).long()
        elif filter_method == 'not_eye':
            filter_ids = (~torch.eye(obj_num).byte()).view(-1).nonzero().squeeze()
        elif filter_method == 'tri_u':
            filter_ids = torch.ones(obj_num, obj_num).triu(diagonal=1).byte().view(-1).nonzero().squeeze()
        elif filter_method == 'tri':
            filter_ids = torch.ones(obj_num, obj_num).triu(diagonal=0).byte().view(-1).nonzero().squeeze()
        else:
            raise NotImplementedError(f'Unknown method {filter_method}')
        return filter_ids.cuda(device)

    @classmethod
    def check_flag(cls, filter_method):
        if filter_method == 'full':
            mirror_flage, loop_flag = True, True
        elif filter_method == 'not_eye':
            mirror_flage, loop_flag = True, False
        elif filter_method == 'tri_u':
            mirror_flage, loop_flag = False, False
        elif filter_method == 'tri':
            mirror_flage, loop_flag = False, True
        else:
            raise NotImplementedError()
        return mirror_flage, loop_flag


class EdgeAddLoop(EdgeOp):
    def __init__(self, last_op, name=None):
        name = name or 'add_loop'
        self.is_func = None
        super().__init__(name, last_op)

    def op_process(self):
        last_op = self.last_op
        if last_op.is_loop:
            self.indexes = last_op.indexes
            self.is_func = False
        else:
            dtype, device = last_op.coords.dtype, last_op.coords.device
            loop = torch.arange(0, last_op.batch_num * last_op.node_num, dtype=dtype, device=device)
            loop = loop.unsqueeze(0).repeat(2, 1)
            self.coords = torch.cat([last_op.coords, loop], dim=1)
            self.is_func = True
        self._loop_flag = True

    def _attr_process(self, attr: torch.Tensor, default_value=1):
        if not self.is_func:
            return attr
        loop_attr = attr.new_full((self.batch_num*self.node_num,), 1)
        if attr.dim() == 2:
            _, k_size = attr.shape
            loop_attr = loop_attr[:, None].repeat(1, k_size)
        return torch.cat((attr, loop_attr), dim=0)


class EdgeRemoveLoop(EdgeOp):
    def __init__(self, last_op, name=None):
        name = name or 'remove_loop '
        self.mask = None
        super().__init__(name, last_op)

    def op_process(self):
        if not self.last_op.is_loop:
            self.indexes = self.last_op.indexes
        else:
            row, col = self.last_op.coords
            mask = row != col
            self.mask = mask
            mask = mask.unsqueeze(0).expand_as(self.last_op.coords)
            self.coords = self.last_op.coords[mask]
        self._loop_flag = False

    def _attr_process(self, attr: torch.Tensor):
        if self.mask is None:
            return attr
        attr = attr.view(-1, attr.shape[-1])
        return attr[self.mask]


class EdgeMirror(EdgeOp):
    op_name = 'MIRROR'

    def __init__(self, last_op, name=None):
        self.sort_idx = None
        name = name or 'mirror'
        super().__init__(name, last_op)

    def op_process(self):
        last_op = self.last_op
        if last_op.is_mirror:
            self._indexes = last_op.indexes
            self.sort_idx = None
        else:
            edge_row, edge_col = last_op.coords
            new_edge_row, new_edge_col = torch.cat((edge_row, edge_col)), torch.cat((edge_col, edge_row))
            edge_ids = self.coord2ids(torch.stack((new_edge_row, new_edge_col)), last_op.node_num)
            edge_ids, sort_idx = edge_ids.sort()
            self.sort_idx = sort_idx  # n
            self.indexes = edge_ids
        self._mirror_flag = True
        self._loop_flag = last_op.is_loop

    def _attr_process(self, attr: torch.Tensor):
        """
        Assuming edge_attr has been processed by last op
        :param attr: m, c
        :return:
        """
        if attr.dim() > 2:
            attr = attr.view(-1, attr.shape[-1])
        assert attr.shape[0] == self.last_op.edge_num
        if self.sort_idx is not None:
            return torch.cat((attr,) * 2, dim=0)[self.sort_idx]
        return attr


class EdgeTopK(EdgeOp):
    op_name = 'TOPK'

    def __init__(self, by_attr: torch.Tensor, reduce_size, last_op, name=None, keep_self=False):
        self.by_attr = by_attr
        self.reduce_size = reduce_size
        self.top_ids = None
        self.keep_self = keep_self
        name = name or f'top_{reduce_size}'
        super().__init__(name, last_op)

    def op_process(self):
        assert self.last_op.is_mirror
        attr, last_op = self.by_attr, self.last_op
        if attr.dim() == 1:
            attr = attr.view(-1, 1)
        attr_c = attr.shape[-1]
        assert attr.view(-1, attr_c).shape[0] == self.last_op.edge_num
        attr = self.reshape(attr, dim_is_4=True)
        self.by_attr, top_ids = self.attr_topk(attr, -2, self.reduce_size, keep_self=self.keep_self)  # b, n_num, k, 1
        self.top_ids = top_ids.squeeze(-1)  # b, n_num, k
        self.indexes = self.reshape(last_op.indexes).gather(index=self.top_ids, dim=2).view(-1)

    @classmethod
    def attr_topk(cls, attr, dim, reduce_size=-1, use_abs=False, keep_self=False):
        """

        :param attr: b_num, n_num, -1, k_num or b_num, n_num, k_num
        :param dim:
        :param reduce_size:
        :param use_abs:
        :return: o_b, n_num, k,
        """
        k_size = attr.shape[-1]
        if reduce_size != -1 and k_size > 1:
            attr = attr.mean(dim=-1, keepdim=True)  # o_b,**, 1
        if reduce_size != -1:
            if use_abs:
                _, top_indexes = attr.abs().topk(reduce_size, dim=dim, sorted=False)
                attr = attr.gather(index=top_indexes, dim=-2)
            else:
                if not keep_self:
                    attr, top_indexes = attr.topk(reduce_size, dim=dim, sorted=False)  # b,obj_num,max_size,k_size
                else:
                    assert attr.size(1) == attr.size(2)
                    loop_mask = torch.eye(attr.size(1))[None, :, :, None].expand(attr.size(0), -1, -1, attr.size(-1)).cuda(attr.device).bool()
                    fake_attr = attr.masked_fill(loop_mask, 1000.)
                    _, top_indexes = fake_attr.topk(reduce_size, dim=dim, sorted=False)
                    attr = attr.gather(index=top_indexes, dim=-2)

        else:
            top_indexes = None
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
        super().__init__(init_method, node.batch_num, node.node_num, node.device)

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

    def spatial_feats(self):
        node_size, node_centre = self.node.spatial_attr
        node_dists = node_intersect(self.node.coords, 'minus')  # b, n, n, 4
        node_dists = node_dists / torch.cat((node_size, node_size), dim=-1).unsqueeze(dim=2)
        node_scale = node_intersect(node_size, 'divide')
        node_mul = node_intersect(node_size[:, :, 0].unsqueeze(-1)*node_size[:, :, 1].unsqueeze(-1), 'divide')
        node_sum = node_intersect(node_size[:, :, 0].unsqueeze(-1)+node_size[:, :, 1].unsqueeze(-1), 'divide')
        return torch.cat((node_dists, node_scale, node_mul, node_sum), dim=-1)









