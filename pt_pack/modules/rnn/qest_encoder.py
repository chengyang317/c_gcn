# coding=utf-8
import torch.nn as nn
from .utils import get_rnn_net
from torch.nn.init import kaiming_uniform_
import logging

logger = logging.getLogger(__name__)


__all__ = ['QestEncoder']


class QestEncoder(nn.Module):
    def __init__(self, embed_opt=None, encoder_opt=None, need_embed=True):
        """

        :param embed_opt: {token_num, word_dim}
        :param encoder_opt:
        :param need_embed:
        """
        super().__init__()
        self.need_embed = need_embed
        if need_embed:
            self.embed_l = nn.Embedding(embed_opt['token_num'], embed_opt['word_dim'])
        if 'input_size' not in encoder_opt['cls_opt']:
            encoder_opt['cls_opt']['input_size'] = embed_opt['word_dim']
        self.encoder_l = get_rnn_net(encoder_opt)
        # self.encoder_l.flatten_parameters()

    def init_parameters(self):
        logger.info(f"Net {self.__class__} is initializing its parameters")
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                kaiming_uniform_(module.weight)

    def forward(self, q, q_len):
        if self.need_embed:
            q = self.embed_l(q)  # b, token_num, word_dim
        out, _ = self.encoder_l(q)
        q_feat = out.gather(1, (q_len-1).view(-1, 1, 1).expand(-1, -1, out.size(-1))).squeeze()
        return q_feat


class QEncoder(nn.Module):
    def __init__(self,
                 embed_opt,
                 encoder_opt,
                 need_embed=True
                 ):
        super().__init__()
        self.need_embed = need_embed
        if need_embed:
            embed_layers = []
            embed_layers.append(nn.Embedding(embed_opt['token_num'], embed_opt['word_dim']))
            if 'dropout' in embed_opt and embed_opt['dropout'] is not None:
                embed_layers.append(nn.Dropout(embed_opt['dropout']))
            self.embed_n = nn.Sequential(*embed_layers)

        if 'input_size' not in encoder_opt['cls_opt']:
            encoder_opt['cls_opt']['input_size'] = embed_opt['word_dim']
        self.encoder_n = get_rnn_net(encoder_opt)

    def forward(self, question, q_len):
        if self.need_embed:
            question = self.embed_n(question)  # b, k, 300
        packed_embed = pack_padded_sequence(question, q_len, batch_first=True)
        packed_out, (last_h, last_c) = self.encoder_n(packed_embed)  # last_h: 2, b, dim
        q_feat = last_h.permute(1, 0, 2).contiguous().view(question.shape[0], -1)  # b, out_dim
        q_ctx, _ = pad_packed_sequence(packed_out, batch_first=True)
        return q_feat, q_ctx

    def init_parameters(self):
        logger.info(f"Net {self.__class__} is initializing its parameters")
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                # nn.init.xavier_normal_(module.bias)





