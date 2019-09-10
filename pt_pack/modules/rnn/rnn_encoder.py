# coding=utf-8
import torch.nn as nn
import logging
from .atten_encoder import SeqAttenEncoder


logger = logging.getLogger(__name__)

__all__ = ['SeqEncoder']


rnn_type_map = {
    'gru': nn.GRU,
    'seq_attn': SeqAttenEncoder
}



class SeqEncoder(nn.Module):
    def __init__(self,
                 embed_num,
                 embed_dim,
                 encode_type,
                 out_dim,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.
                 ):

        super().__init__()
        self.embed_l = nn.Embedding(embed_num, embed_dim)
        assert encode_type in rnn_type_map
        self.encoder_l = rnn_type_map[encode_type](embed_dim, out_dim, num_layers, bidirectional=bidirectional,
                                                   dropout=dropout, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        logger.info(f'Net {self.__class__} is resetting its parameters')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.)

    def forward(self, seq, seq_len):
        seq_embed = self.embed_l(seq)
        seq_out, _ = self.encoder_l(seq_embed)
        seq_feat = seq_out.gather(1, (seq_len-1).view(-1, 1, 1).expand(-1, -1, seq_out.size(-1))).squeeze()
        return seq_feat