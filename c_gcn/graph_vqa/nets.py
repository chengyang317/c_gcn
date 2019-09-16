# coding=utf-8
from pt_pack import LayerNet, Net, Dataset
from pt_pack.utils import try_set_attr, load_func_params, masked_softmax
from .graph import Graph
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch


__all__ = ['CondGraphVqaNet', 'CgsGraphQNet', 'ResGraphVqaNet', 'CondGraphQNet', 'AttnGraphQNet', 'ModuleGraphQNet']


class CgsGraphQNet(Net):
    prefix = 'graph_q_net'

    def __init__(self, q_vocab, embed_dim: int = 300, hid_dim: int = 1024, dropout: float = 0.):
        super().__init__()
        self.q_vocab = q_vocab
        self.embedding = nn.Embedding(len(q_vocab), embed_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True)
        self.dropout_l = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        super().reset_parameters()
        print('glove init')
        self.embedding.weight.data.copy_(self.q_vocab.glove_embed('glove.6B.300d'))

    def forward(self, q_labels, q_len):
        emb = self.embedding(q_labels)
        packed = pack_padded_sequence(emb, q_len.squeeze().tolist(), batch_first=True)
        _, hid = self.rnn(packed)
        hid = self.dropout_l(hid)
        return hid.squeeze()

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        dataset_cls = Dataset.load_cls(params.dataset_cls)
        q_vocab, _ = dataset_cls.load_combined_vocab()
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        kwargs[f'{cls.prefix_name()}_q_vocab'] = q_vocab
        return cls.default_build(kwargs, controller=controller)


class ModuleGraphQNet(Net):
    prefix = 'graph_q_net'

    def __init__(self,
                 q_vocab,
                 module_num: int = 3,
                 embed_dim: int = 300,
                 hid_dim: int = 2048,
                 out_dim: int = 1024,
                 dropout: float = 0.):
        super().__init__()
        self.q_vocab = q_vocab
        self.embedding = nn.Embedding(len(q_vocab), embed_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True)
        self.linear_l = nn.utils.weight_norm(nn.Linear(hid_dim, module_num * out_dim))
        self.dropout_l = nn.Dropout(dropout)
        self.module_num = module_num
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        super().reset_parameters()
        print('glove init')
        self.embedding.weight.data.copy_(self.q_vocab.glove_embed('glove.6B.300d'))

    def forward(self, q_labels, q_len):
        emb = self.embedding(q_labels)
        packed = pack_padded_sequence(emb, q_len.squeeze().tolist(), batch_first=True)
        _, hid = self.rnn(packed)
        out = self.linear_l(hid.squeeze())
        out = self.dropout_l(out)
        return out.chunk(self.module_num, dim=-1)

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        dataset_cls = Dataset.load_cls(params.dataset_cls)
        q_vocab, _ = dataset_cls.load_combined_vocab()
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        kwargs[f'{cls.prefix_name()}_q_vocab'] = q_vocab
        return cls.default_build(kwargs, controller=controller)


class CondGraphQNet(Net):
    prefix = 'graph_q_net'

    def __init__(self, q_vocab, embed_dim: int = 300, hid_dim: int = 1024, dropout: float = 0.):
        super().__init__()
        self.q_vocab = q_vocab
        self.embedding = nn.Embedding(len(q_vocab), embed_dim)
        self.rnn = nn.LSTM(embed_dim, hid_dim//2, batch_first=True, bidirectional=True)
        self.dropout_l = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        self.embedding.weight.data.copy_(self.q_vocab.glove_embed('glove.6B.300d'))

    def forward(self, q_labels, q_len):
        emb = self.embedding(q_labels)
        packed = pack_padded_sequence(emb, q_len.squeeze().tolist(), batch_first=True)
        _, hid = self.rnn(packed)
        hid = hid[0].transpose(0, 1).contiguous().view(emb.size(0), -1)
        return self.dropout_l(hid)

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        dataset_cls = Dataset.load_cls(params.dataset_cls)
        q_vocab, _ = dataset_cls.load_combined_vocab()
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        kwargs[f'{cls.prefix_name()}_q_vocab'] = q_vocab
        return cls.default_build(kwargs, controller=controller)


class AttnGraphQNet(Net):
    prefix = 'graph_q_net'

    def __init__(self,
                 q_vocab,
                 module_num: int = 3,
                 embed_dim: int = 300,
                 mlp_dim: int = 512,
                 hid_dim: int = 1024,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.q_vocab = q_vocab
        self.module_num = module_num
        self.mlp_dim = mlp_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(len(q_vocab), embed_dim)
        self.mlp_l = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # self.rnn_l = nn.LSTM(mlp_dim, hid_dim//2, 1, batch_first=True, bidirectional=True)
        self.rnn_l = nn.GRU(mlp_dim, hid_dim, batch_first=True)
        self.attn_list = nn.ModuleList([nn.Linear(hid_dim, 1) for _ in range(self.module_num)])
        # self.dropout_l = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self, init='uniform'):
        super().reset_parameters()
        print('glove init')
        self.embedding.weight.data.copy_(self.q_vocab.glove_embed('glove.6B.300d'))

    def forward(self, q_labels, q_len):
        b_num, len_num = q_labels.shape
        emb = self.embedding(q_labels)
        # emb = self.dropout_l(emb)
        emb = self.mlp_l(emb)

        packed = pack_padded_sequence(emb, q_len.squeeze().tolist(), batch_first=True)
        output, _ = self.rnn_l(packed)
        ctx = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=len_num)[0]  # b, l, h
        # hid = hidden[0].transpose(0, 1).contiguous().view(emb.size(0), -1)  # b, h
        mask = torch.arange(len_num).expand(b_num, len_num).cuda(q_labels.device) >= q_len.unsqueeze(-1)
        attn_embs = list()
        for idx in range(self.module_num):
            ctx_score = self.attn_list[idx](ctx).squeeze()  # b, l
            attn = masked_softmax(ctx_score, mask)
            attn_emb = torch.bmm(attn.unsqueeze(1), emb).squeeze()
            attn_embs.append(attn_emb)
        return attn_embs

    @classmethod
    def build(cls, params, sub_cls=None, controller=None):
        dataset_cls = Dataset.load_cls(params.dataset_cls)
        q_vocab, _ = dataset_cls.load_combined_vocab()
        kwargs = load_func_params(params, cls.__init__, cls.prefix_name())
        kwargs[f'{cls.prefix_name()}_q_vocab'] = q_vocab
        return cls.default_build(kwargs, controller=controller)


class CondGraphVqaNet(LayerNet):
    prefix = 'graph_vqa_net'

    def __init__(self,
                 layers,
                 filter_method: str = 'tri_u',
                 ):
        super().__init__(layers)
        self.filter_method = filter_method
        self.reset_parameters()

    def forward(self, obj_feats, obj_coord, q_feats):
        graph = Graph(obj_feats, obj_coord, filter_method=self.filter_method)
        conv_id = 0
        for layer_id, layer in enumerate(self.layers):
            if not isinstance(q_feats, (list, tuple)):
                graph.cond_feats = q_feats
            else:
                if 'conv' in layer.__class__.__name__.lower():
                    graph.cond_feats = q_feats[conv_id]
                    conv_id += 1
            graph.edge.clear_ops()
            graph = layer(graph)
        return graph

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_layers', ('sparse_graph_layer',))
        try_set_attr(params, f'{cls.prefix_name()}_layer_obj_dims', (2048,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_q_dims', (1024,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_out_dims', (1024,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_kernel_sizes', (8,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_reduce_sizes', (16,))
        try_set_attr(params, f'{cls.prefix_name()}_filter_method', 'not_eye')


class ResGraphVqaNet(LayerNet):
    prefix = 'graph_vqa_net'

    def __init__(self,
                 layers,
                 filter_method: str = 'tri_u',
                 ):
        super().__init__(layers)
        self.filter_method = filter_method

    def forward(self, obj_feats, obj_coord, q_feats):
        graph = Graph(obj_feats, obj_coord, cond_feats=q_feats, filter_method=self.filter_method)
        for layer_id, layer in enumerate(self.layers):
            graph.edge.clear_ops()
            graph = layer(graph)
            if layer_id in (0, len(self.layers)-1):
                continue
            graph.cond_feats = graph.graph_feats('last') + graph.cond_feats
            graph.feats[-1] = graph.graph_feats('last') + graph.cond_feats
        return graph

    @classmethod
    def init_args(cls, params, sub_cls=None):
        try_set_attr(params, f'{cls.prefix_name()}_layers', ('sparse_graph_layer',))
        try_set_attr(params, f'{cls.prefix_name()}_layer_obj_dims', (2048,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_q_dims', (1024,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_out_dims', (1024,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_kernel_sizes', (8,))
        try_set_attr(params, f'{cls.prefix_name()}_layer_reduce_sizes', (16,))
        try_set_attr(params, f'{cls.prefix_name()}_filter_method', 'not_eye')















