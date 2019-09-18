# coding=utf-8
import torch
import math


__all__ = ['to_edge_idx', 'tri_mask', 'tri_index', 'is_same', 'node_intersect', 'edge_filter', 'edge_select', 'is_nan',
           'is_inf']


def to_edge_idx(top_id, k=None):
    """

    :param var: b, k, c, usually top_id
    :return:
    """

    if top_id.dim() == 2:
        b, c = top_id.shape
        b_prefix = torch.arange(0, b) * k
        b_prefix = b_prefix.type_as(top_id).view(b, 1)
        row = top_id // k + b_prefix
        col = top_id % k + b_prefix
    else:
        b, k, c = top_id.shape
        b_prefix = torch.arange(0, b) * k
        b_prefix = b_prefix.type_as(top_id).view(b, 1, 1)
        col = top_id + b_prefix  # b, k, c
        row = torch.arange(0, k).view(1, -1, 1).type_as(top_id) + b_prefix
        row = row.repeat(1, 1, c)
    return torch.stack((row.view(-1), col.view(-1)), dim=0)


def tri_mask(k, diagonal=0, up=True):
    if up:
        return torch.ones(k, k).triu(diagonal=diagonal).byte()
    return torch.ones(k, k).tril(diagonal=diagonal).byte()


def tri_index(k, diagonal=0, up=True):
    mask = tri_mask(k, diagonal, up)
    return mask.view(-1).nonzero().squeeze()


def is_same(s, t):
    res = s != t
    res = res.sum().item()
    return False if res != 0 else True


def node_intersect(obj_feats: torch.Tensor, method = 'sum'):
    """

    :param obj_feats: b,k,c
    :return:
    """
    # assert obj_feats.dim() == 3 and method in ('sum', 'cat', 'mul', 'inter')
    x_row = obj_feats[:, :, None, :]
    x_col = obj_feats[:, None, ::]
    if method == 'sum':
        return x_row + x_col
    elif method == 'minus':
        return x_row - x_col
    elif method == 'mul':
        return x_row * x_col
    elif method == 'divide':
        return x_row / x_col
    elif method == 'max':
        return torch.max(x_row, x_col)
    elif method == 'cat':
        _, k, _ = obj_feats.shape
        return torch.cat((x_row.expand(-1, -1, k, -1), x_col.expand(-1, k, -1, -1)), dim=-1)
    elif method == 'inter':
        _, k, _ = obj_feats.shape
        A = torch.cat((x_row.expand(-1, -1, k, -1), x_col.expand(-1, k, -1, -1)), dim=-1)
        B = torch.cat((x_col.expand(-1, k, -1, -1), x_row.expand(-1, -1, k, -1)), dim=-1)
        return A, B
    else:
        raise NotImplementedError()


def edge_filter(edge_feats: torch.Tensor, method='full'):
    """

    :param edge_feats: b,k,k,c
    :param method:
    :return:
    """
    assert edge_feats.dim() == 4 and method in ('full', 'not_eye', 'tri_u')
    b, k, _, _ = edge_feats.shape
    if method == 'full':
        return edge_feats.view(b, k * k, -1), torch.arange(0, k * k, device=edge_feats.device).long()
    elif method == 'not_eye':
        filter_ids = (~torch.eye(k, device=edge_feats.device).byte()).view(-1).nonzero().squeeze()
        return edge_feats.view(b, k * k, -1).index_select(1, filter_ids), filter_ids
    elif method == 'tri_u':
        filter_ids = torch.ones(k, k, device=edge_feats.device).triu(diagonal=1).byte().view(-1).nonzero().squeeze()
        return edge_feats.view(b, k * k, -1).index_select(1, filter_ids), filter_ids
    else:
        raise NotImplementedError()


def edge_ids_dissovle_batch(edge_ids: torch.Tensor, k):
    b, c = edge_ids.shape
    b_prefix = torch.arange(0, b) * k
    b_prefix = b_prefix.type_as(edge_ids).view(b, 1)
    row = edge_ids // k + b_prefix
    col = edge_ids % k + b_prefix
    return torch.stack((row.view(-1), col.view(-1)), dim=0)


def edge_select(edge_prob: torch.Tensor, edge_num, filter_ids, method='full'):
    assert edge_prob.dim() == 2 and method in ('full', 'not_eye', 'tri_u') and filter_ids.dim() == 1
    b, m = edge_prob.shape
    if method == 'full':
        raise NotImplementedError()
    elif method == 'not_eye':
        k = math.ceil(math.sqrt(m))
        top_k, top_id = edge_prob.view(b, k, k-1).topk(edge_num, dim=-1, sorted=False)  # b,k,n
        edge_weight = top_k.softmax(dim=-1)  # b, k, n
        edge_ids = filter_ids[None].expand(b, -1).view(b, k, k-1).gather(index=top_id, dim=-1).view(b, -1)
    elif method == 'tri_u':
        k = math.ceil(math.sqrt(m*2))
        top_k, top_id = edge_prob.topk(edge_num, dim=-1, sorted=False)  # b, n
        edge_weight = top_k.softmax(dim=-1)  # b, n
        edge_ids = filter_ids[None, :].expand(b, -1).gather(index=top_id, dim=-1)  # b, n
    else:
        raise NotImplementedError()
    edge_ids = edge_ids_dissovle_batch(edge_ids, k)
    return edge_ids, edge_weight.view(-1)


def is_nan(x):
    return True if torch.isnan(x).sum() > 0 else False


def is_inf(x):
    return True if torch.isinf(x).sum() > 0 else False


if __name__ == '__main__':
    a = torch.rand(2, 5, 5)
    _, edge_ids = a.topk(3, dim=-1, sorted=False)
    edge_indx = to_edge_idx(edge_ids)


