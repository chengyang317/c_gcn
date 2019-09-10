# coding=utf-8
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


__all__ = ['SeqConvEncoder']



def Embedding(embed_num, embed_dim, padding_idx=None):
    m = nn.Embedding(embed_num, embed_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad
        self.range_buf = torch.arange(num_embeddings)

    def forward(self, seq_tokens, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = self.range_buf[:seq_tokens.size(-1)].expand_as(seq_tokens).type_as(seq_tokens)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1


def PositionalEmbedding(embed_num, embed_dim, padding_idx, left_pad=False):
    m = LearnedPositionalEmbedding(embed_num, embed_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


def ConvTBC(in_dim, out_dim, kernel_size, dropout=0., **kwargs):
    """Weight-normalized Conv1d layer"""
    from .conv_tbc import ConvTBC
    m = ConvTBC(in_dim, out_dim, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_dim))
    nn.init.normal_(m.weight, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SeqConvEncoder(nn.Module):
    def __init__(self,
                 embed_num,
                 embed_dim,
                 post_embed_num,
                 padding_idx=0,
                 left_pad=False,
                 out_dim=512,
                 norm_cons=0.5,
                 dropout=0.2,
                 ):
        super().__init__()
        conv_config = [(out_dim, 3, 1)] * 4
        self.dropout = dropout
        self.conv_config = conv_config
        self.norm_cons = norm_cons
        self.embed_l = Embedding(embed_num, embed_dim, padding_idx)
        self.post_embed_l = PositionalEmbedding(post_embed_num, embed_dim, None, left_pad)

        conv_in_dim = conv_config[0][0]
        self.fc1_l = Linear(embed_dim, conv_in_dim, dropout)

        self.proj_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.residual_idxs = []

        conv_in_dims = [conv_in_dim]
        for i, (conv_out_dim, kernel_size, residual_idx) in enumerate(conv_config):
            if residual_idx == 0:
                residual_dim = conv_out_dim
            else:
                residual_dim = conv_in_dims[-residual_idx]
            self.proj_layers.append(Linear(residual_dim, conv_out_dim) if residual_dim != conv_out_dim else None)
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.conv_layers.append(
                ConvTBC(conv_in_dim, conv_out_dim * 2, kernel_size, dropout=dropout, padding=padding)
            )
            self.residual_idxs.append(residual_idx)
            conv_in_dim = conv_out_dim
            conv_in_dims.append(conv_out_dim)

        # experiments:
        # self.seq2one_l = nn.GRU(conv_out_dim, conv_out_dim)

        self.fc2_l = Linear(conv_in_dim, embed_dim)
        self.fc3_l = nn.Linear(post_embed_num, 1)

    def forward(self, seq_tokens, seq_len):
        padding_mask = torch.arange(seq_tokens.shape[1]).expand_as(seq_tokens).type_as(seq_len).ge(seq_len.unsqueeze(1))
        embed = self.embed_l(seq_tokens) + self.post_embed_l(seq_tokens)
        embed = embed.masked_fill(padding_mask.unsqueeze(-1), 0)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        # project to size of convolution
        seq_feat = self.fc1_l(embed).transpose(0, 1)  # t, b, c

        padding_mask = padding_mask.t() if padding_mask.any() else None  # t,b
        residuals = [seq_feat]
        # temporal convolutions
        for proj_l, conv_l, res_idx in zip(self.proj_layers, self.conv_layers, self.residual_idxs):
            if res_idx > 0:
                residual = residuals[-res_idx]
                residual = residual if proj_l is None else proj_l(residual)
            else:
                residual = None

            if padding_mask is not None:
                seq_feat = seq_feat.masked_fill(padding_mask.unsqueeze(-1), 0)

            seq_feat = F.dropout(seq_feat, p=self.dropout, training=self.training)
            if conv_l.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                seq_feat = conv_l(seq_feat)
            else:
                padding_l = (conv_l.kernel_size[0] - 1) // 2
                padding_r = conv_l.kernel_size[0] // 2
                seq_feat = F.pad(seq_feat, (0, 0, 0, 0, padding_l, padding_r))
                seq_feat = conv_l(seq_feat)
            seq_feat = F.glu(seq_feat, dim=2)

            if residual is not None:
                seq_feat = (seq_feat + residual) * math.sqrt(self.norm_cons)
            residuals.append(seq_feat)

        # seq_out, _ = self.seq2one_l(seq_feat)
        # seq_out = seq_out.transpose(1, 0)
        # seq_feat = seq_out.gather(1, (seq_len-1).view(-1, 1, 1).expand(-1, -1, seq_out.size(-1))).squeeze()
        # T x B x C -> B x T x C
        seq_feat = seq_feat.transpose(1, 0)

        # project back to size of embedding
        # seq_feat_embed = self.fc2_l(seq_feat)
        #
        # # scale gradients (this only affects backward, not forward)
        # # x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))
        #
        # # add output to input embedding for attention
        # seq_ctx = (seq_feat_embed + embed) * math.sqrt(self.norm_cons)
        #
        # if padding_mask is not None:
        #     padding_mask = padding_mask.t()  # -> B x T
        #     # seq_feat = seq_feat.masked_fill(padding_mask.unsqueeze(-1), 0)
        #     seq_ctx = seq_ctx.masked_fill(padding_mask.unsqueeze(-1), 0)
        seq_ctx = None
        return seq_feat.max(dim=1)[0]












