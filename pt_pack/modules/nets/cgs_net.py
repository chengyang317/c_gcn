# coding=utf-8
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from .base import Net

__all__ = ['CgsQNet', 'CgsClsNet']


class CgsQNet(Net):
    prefix = 'q_net'

    def __init__(self, vocab, embed_dim: int = 300, hid_dim: int = 1024, dropout: float = 0.):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(vocab.glove_embed('glove.6B.300d')))
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True)
        self.dropout_l = nn.Dropout(dropout)

    def forward(self, q_labels, q_len):
        emb = self.embedding(q_labels)
        packed = pack_padded_sequence(emb, q_len.squeeze().tolist(), batch_first=True)
        _, hid = self.rnn(packed)
        hid = self.dropout_l(hid)
        return hid.squeeze()

    @classmethod
    def build(cls, params, sub_cls=None):
        setattr(params, f'{cls.prefix_name()}_vocab', params.datasets[0].question_vocab)
        return cls.default_build(params)


class NeighbourOperater(object):
    __slots__ = ['adjacency_matrix', 'nh_size', 'top_k', 'top_id']

    def __init__(self, adjacency_matrix, nh_size):
        self.adjacency_matrix = adjacency_matrix
        self.nh_size = nh_size
        top_k, top_id = adjacency_matrix.topk(nh_size, dim=-1, sorted=False)
        self.top_k, self.top_id = top_k.softmax(dim=-1), top_id  # b, k, nh_size

    def __call__(self, obj_feat, pseudo_coord, is_weight=True):
        """

        :param obj_feat: b, k, c
        :param pseudo_coord: b, k, k, 2
        :param is_weight:
        :return:
        """
        B, K, C = obj_feat.shape
        top_id = self.top_id.unsqueeze(-1).expand(B, K, self.nh_size, C)
        obj_feat = obj_feat.unsqueeze(1).expand(B, K, K, C)

        obj_nh_feat = obj_feat.gather(2, index=top_id)  # b, k, nh_size, c
        if is_weight:
            obj_nh_feat = self.top_k.unsqueeze(-1) * obj_nh_feat

        top_id = self.top_id.unsqueeze(-1).expand(B, K, self.nh_size, 2)
        obj_nh_coord = pseudo_coord.gather(2, index=top_id)
        return obj_nh_feat, obj_nh_coord


# class CgsGraphNet(Net):
#     # prefix = 'graph_net'
#
#     def __init__(self,
#                  obj_dim: int=2052,
#                  q_dim: int=1024,
#                  out_dim: int=1024,
#                  nh_size: int=16,
#                  kernel_num: int=8,
#                  dropout: float=0.5,
#
#                  ):
#         super().__init__()
#
#         self.graph_learner_l = CgsGraphLearnerLayer(obj_dim + q_dim, hid_dim=512)
#         self.graph_convs = nn.ModuleList(
#             [CgsGraphConvLayer(obj_dim, out_dim * 2, kernel_num), CgsGraphConvLayer(out_dim * 2, out_dim, kernel_num)]
#         )
#
#         self.nh_size = nh_size
#         # self.dropout = nn.Dropout(dropout)
#
#     def forward(self, obj_feat, q_feat):
#         B, K, _ = obj_feat.shape
#         pseudo_coord = self.compute_pseudo(obj_feat[:, :, -4:])
#         # obj_feat = self.dropout(obj_feat)
#         q_feat = q_feat.unsqueeze(1).repeat(1, K, 1)  # b, k, _
#         adjacency_matrix = self.graph_learner_l(torch.cat((obj_feat, q_feat), dim=-1))  # b, k, k
#         if torch.isnan(adjacency_matrix).sum().item() > 0:
#             print('bbbbb')
#         nh_operator = NeighbourOperater(adjacency_matrix, self.nh_size)
#         for idx, conv_layer in enumerate(self.graph_convs):
#             obj_nh_feat, obj_nh_coord = nh_operator(obj_feat, pseudo_coord, is_weight=True if idx==0 else False)
#             obj_feat = conv_layer(obj_nh_feat, obj_nh_coord)
#             # obj_feat = self.dropout(obj_feat)
#         feat = obj_feat.max(dim=1)[0]
#         if torch.isnan(feat).sum().item() > 0:
#             print('xxxxx')
#         return feat
#
#     def compute_pseudo(self, b_box):
#         bb_size = (b_box[:, :, 2:] - b_box[:, :, :2])
#         bb_centre = b_box[:, :, :2] + 0.5 * bb_size  # b, k, 2
#         K = bb_centre.size(1)
#         # Compute cartesian coordinates (batch_size, K, K, 2)
#         pseudo_coord = bb_centre.view(-1, K, 1, 2) - bb_centre.view(-1, 1, K, 2)
#         # Conver to polar coordinates
#         rho = torch.sqrt(pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
#         theta = torch.atan2(pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
#         pseudo_coord = torch.cat((torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)
#         return pseudo_coord


class CgsClsNet(Net):
    prefix = 'cls_net'

    def __init__(self, in_dim: int=1024, out_dim: int=3001, dropout: float=0.5):
        super().__init__()
        self.linear_0 = nn.utils.weight_norm(nn.Linear(in_dim, out_dim))
        self.linear_1 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))
        self.relu_l = nn.ReLU()

    def forward(self, img_feat, q_feat):
        feat = img_feat * self.relu_l(q_feat)
        feat = self.linear_0(feat)
        logits = self.linear_1(feat)
        return logits
