# coding=utf-8
import torch.nn as nn
from ..fair_seq.multihead_attention import MultiheadAttention
from ..fair_seq.transformer import Embedding, Linear, LayerNorm, PositionalEmbedding, SinusoidalPositionalEmbedding
import torch.nn.functional as F
import math


__all__ = ['SeqAttenEncoder']


class AttenEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim,
                 ffn_embed_dim,
                 dropout=0.2,
                 relu_dropout=0.,
                 attention_headers=4,
                 attention_dropout=0.,
                 is_norm_before=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, attention_headers, attention_dropout)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.norm_before = is_norm_before
        self.fc1_l = Linear(embed_dim, ffn_embed_dim)
        self.fc2_l = Linear(ffn_embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, seq_feat, padding_mask):
        """

        :param seq_feat: T, B, C
        :param padding_mask: B, T
        :return:
        """
        residual = seq_feat
        seq_feat = self.maybe_layer_norm(0, seq_feat, before=True)
        seq_feat, _ = self.self_attn(query=seq_feat, key=seq_feat, value=seq_feat, key_padding_mask=padding_mask)
        seq_feat = F.dropout(seq_feat, p=self.dropout, training=self.training)
        seq_feat = residual + seq_feat
        seq_feat = self.maybe_layer_norm(0, seq_feat, after=True)

        residual = seq_feat
        seq_feat = self.maybe_layer_norm(1, seq_feat, before=True)
        seq_feat = F.relu(self.fc1_l(seq_feat))
        seq_feat = F.dropout(seq_feat, p=self.relu_dropout, training=self.training)
        seq_feat = self.fc2_l(seq_feat)
        seq_feat = F.dropout(seq_feat, p=self.dropout, training=self.training)
        seq_feat = residual + seq_feat
        seq_feat = self.maybe_layer_norm(1, seq_feat, after=True)
        return seq_feat

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.norm_before:
            return self.layer_norms[i](x)
        else:
            return x


class SeqAttenEncoder(nn.Module):
    """Seq attention encoder."""

    def __init__(self,
                 embed_num,
                 embed_dim,
                 post_embed_num,
                 ffn_embed_dim=256,
                 padding_idx=0,
                 left_pad=False,
                 dropout=0.2,
                 post_embed_learn=False,
                 encoder_layers=3,
                 relu_dropout=0.,
                 attention_headers=4,
                 attention_dropout=0.,
                 is_norm_before=False,
                 ):
        super().__init__()
        self.dropout = dropout
        self.embed_l = Embedding(embed_num, embed_dim, padding_idx)
        self.post_embed_l = PositionalEmbedding(post_embed_num, embed_dim, padding_idx, left_pad, post_embed_learn)
        self.embed_scale = math.sqrt(embed_dim)
        self.padding_idx = padding_idx
        encoder_layer_opt = (embed_dim, ffn_embed_dim, dropout, relu_dropout, attention_headers,
                             attention_dropout, is_norm_before)
        self.encoder_layers = nn.ModuleList([AttenEncoderLayer(*encoder_layer_opt) for _ in range(encoder_layers)])

    def forward(self, seq_tokens, seq_len):
        # embed tokens and positions
        embed = self.embed_l(seq_tokens) * self.embed_scale
        # embed += self.post_embed_l(seq_tokens)
        embed = F.dropout(embed, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        embed = embed.transpose(0, 1)

        # compute padding mask
        padding_mask = seq_tokens.eq(self.padding_idx)  # B, T
        if not padding_mask.any():
            padding_mask = None

        for encoder_layer in self.encoder_layers:
            embed = encoder_layer(embed, padding_mask)  # T, B, C
        seq_feat = embed.transpose(0, 1).contiguous()
        seq_feat = seq_feat.masked_fill(padding_mask.unsqueeze(-1), 0)
        seq_feat = seq_feat.max(dim=1)
        return seq_feat








