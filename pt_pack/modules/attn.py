# coding=utf-8
import torch.nn as nn
from torch.nn import Parameter
import torch
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, in_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = in_dim // num_heads
        assert self.head_dim * num_heads == self.in_dim
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3 * in_dim, in_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * in_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(in_dim, in_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def splitC2B(self, tensor, split_num):
        """
        :param tensor:  shape is B, T, C
        :return:  B*K, T, C/K
        """
        bsz, tsz, csz = tensor.shape
        tensor = tensor.transpose(0, 1).contiguous().view(tsz, int(bsz*split_num), int(csz/split_num))  # T, B*K, C/K
        return tensor.transpose(0, 1).contiguous()  # B*K, T, C/K

    def splitB2C(self, tensor, split_num):
        """
        :param tensor:  shape is B*K, T, C/K
        :return:  B, T, C
        """
        bsz, tsz, csz = tensor.shape
        tensor = tensor.transpose(0, 1).contiguous().view(tsz, int(bsz/split_num), int(csz*split_num))  # T, B, C
        return tensor.transpose(0, 1).contiguous()  # B, T, C

    def forward(self, query, key, value, mask_future_timesteps=False, key_padding_mask=None, need_weights=False):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        bsz, tgt_len, in_dim = query.size()
        assert in_dim == self.in_dim
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q, k, v = [self.splitC2B(x, self.num_heads) for x in (q, k, v)]  # B*K, T, C/K

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # B*k, T, T
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if mask_future_timesteps:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # B*K, T, C/K
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = self.splitB2C(attn, self.num_heads)  # B, T, C
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.in_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.in_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.in_dim, end=2 * self.in_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.in_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
























































