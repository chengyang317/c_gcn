# coding=utf-8
import torch_geometric.nn as gnn
from .layers import AggregateGraphCalcLayer, KernelGraphCalcLayer, LinearGraphCalcLayer, GraphFilm


__all__ = ['build_graph_calc_layer']


def build_graph_calc_layer(graph_method, **kwargs):
    if graph_method == 'aggregate':
        return AggregateGraphCalcLayer()
    elif graph_method == 'kernel':
        return KernelGraphCalcLayer(kwargs['in_dim'], kwargs['out_dim'], kwargs['kernel_size'])
    elif graph_method == 'linear':
        return LinearGraphCalcLayer(kwargs['in_dim'], kwargs['out_dim'])
    elif graph_method == 'cond':
        return GraphFilm(kwargs['in_dim'], kwargs['out_dim'])
    else:
        graph_cls = getattr(gnn, graph_method, None)
        assert graph_cls is not None
        return graph_cls(kwargs['in_dim'], kwargs['out_dim'])