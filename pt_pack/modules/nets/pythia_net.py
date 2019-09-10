# coding=utf-8
from .base import Net
import torch.nn as nn
import torch

__all__ = ['PythiaQNet']


class PythiaQNet(Net):
    prefix = 'q_net'

    def __init__(self, vocab, embed_dim: int = 300, hid_dim: int = 1024):
        super(PythiaQNet, self).__init__()
        self.embed_l = nn.Embedding(len(vocab), embed_dim)
        self.rnn_l = nn.LSTM(input_size=embed_dim, hidden_size=hid_dim, num_layers=1, batch_first=True)

        self.attention_l = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid_dim//2, 2, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        self.hid_dim = hid_dim
        if getattr(vocab, 'glove_embed') is not None:
            self.embed_l.weight.data.copy_(torch.from_numpy(vocab.glove_embed('glove.6B.300d')))

    @property
    def out_dim(self):
        return self.hid_dim * 2

    def forward(self, q_labels, q_lens):
        b, _ = q_labels.shape
        embed_txt = self.embed_l(q_labels)  # b,t,c
        rnn_out, _ = self.rnn_l(embed_txt)  # b, t, c
        rnn_reshape = rnn_out.permute(0, 2, 1)  # b, c, t
        attention = self.attention_l(rnn_reshape)  # b, 2, t
        feat = attention.bmm(rnn_out)  # n, 2, c
        return feat.view(b, -1)  # b,2c

    @classmethod
    def build(cls, params, sub_cls=None):
        setattr(params, f'{cls.prefix_name()}_vocab', params.train_dataset.question_vocab)
        return cls.default_build(params)


