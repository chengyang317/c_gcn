# coding=utf-8
import torch.nn as nn


def get_rnn_net(encoder_opt):
    """

    :param encoder_opt: {cls_name, cls_opt}
    :return:
    """
    cls_name = encoder_opt['cls_name']
    assert hasattr(nn, cls_name)
    return getattr(nn, cls_name)(**encoder_opt['cls_opt'])