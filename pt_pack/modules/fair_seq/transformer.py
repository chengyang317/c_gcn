# coding=utf-8
import torch.nn as nn
import math
import torch.nn.functional as F
from .multihead_attention import MultiheadAttention
import torch
from . import utils


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=embedding_dim ** -0.5)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = utils.make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        # self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            ).type_as(self.weights)
        self.weights = self.weights.to(input.device)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        positions = utils.make_positions(input.data, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

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
        self.linear_1_l = Linear(embed_dim, ffn_embed_dim)
        self.linear_2_l = Linear(ffn_embed_dim, embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, seq_feat, padding_mask):
        residual = seq_feat
        seq_feat = self.maybe_layer_norm(0, seq_feat, before=True)
        seq_feat, _ = self.self_attn(query=seq_feat, key=seq_feat, value=seq_feat, key_padding_mask=padding_mask)
        seq_feat = F.dropout(seq_feat, p=self.dropout, training=self.training)
        seq_feat = residual + seq_feat
        seq_feat = self.maybe_layer_norm(0, seq_feat, after=True)

        residual = seq_feat
        seq_feat = self.maybe_layer_norm(1, seq_feat, before=True)
        seq_feat = F.relu(self.linear_1_l(seq_feat))
        seq_feat = F.dropout(seq_feat, p=self.relu_dropout, training=self.training)
        seq_feat = self.linear_2_l(seq_feat)
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


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self,
                 embed_num,
                 embed_dim,
                 post_embed_num,
                 ffn_embed_dim,
                 padding_idx,
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
        encoder_layer_opt = (embed_dim, ffn_embed_dim, dropout, relu_dropout, attention_headers,
                             attention_dropout, is_norm_before)
        self.encoder_n = nn.Sequential([TransformerEncoderLayer(*encoder_layer_opt) for _ in range(encoder_layers)])

    def forward(self, seq_tokens, seq_len):
        # embed tokens and positions
        embed = self.embed_l(seq_tokens) * self.embed_scale
        embed += self.post_embed_l(seq_tokens)
        embed = F.dropout(embed, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        embed = embed.transpose(0, 1)

        # compute padding mask
        padding_mask = seq_tokens.eq(self.padding_idx)  # B, T
        if not padding_mask.any():
            padding_mask = None

        feat = self.encoder_n(embed, padding_mask)  # T, B, C
        return feat, padding_mask

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            if 'encoder.embed_positions._float_tensor' not in state_dict:
                state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict