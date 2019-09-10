# coding=utf-8
import numpy as np
import torch
from pt_pack import totensor


__all__ = ['base_anchor', 'mk_anchor', 'ratio_enum', 'scale_enum', 'hw_c', 'box_loc', 'box_iou',
           'box_update', 'box_clip', 'box_union']






def base_anchor(base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    base_anchor = np.array([0, 0, base_size, base_size]).reshape((1, 4))
    ratio_anchors = ratio_enum(base_anchor, np.array(ratios))
    anchors = np.vstack([scale_enum(ratio_anchors[i, :], np.array(scales)) for i in range(len(ratio_anchors))])
    return totensor(anchors).float()


def mk_anchor(hs, ws, h_c, w_c):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((w_c - 0.5 * ws, h_c - 0.5 * hs, w_c + 0.5 * ws, h_c + 0.5 * hs))
    return anchors


def ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    h, w, h_c, w_c = hw_c(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = mk_anchor(hs, ws, h_c, w_c)
    return anchors


def scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    h, w, h_c, w_c = hw_c(anchor.reshape(1, 4))
    ws = w * scales
    hs = h * scales
    anchors = mk_anchor(hs, ws, h_c, w_c)
    return anchors


def hw_c(box):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    w_c = box[:, 0] + 0.5 * w
    h_c = box[:, 1] + 0.5 * h
    return h, w, h_c, w_c


def box_update(box, delta):
    w = box[:, :, 2::4] - box[:, :, 0::4]  # b,n,1
    h = box[:, :, 3::4] - box[:, :, 1::4]
    w_c = box[:, :, 0::4] + 0.5 * w
    h_c = box[:, :, 1::4] + 0.5 * h

    dw_c = delta[:, :, 0::4]  # b,n,1
    dh_c = delta[:, :, 1::4]
    dw = delta[:, :, 2::4]
    dh = delta[:, :, 3::4]

    pred_h_c = dh_c * h + h_c
    pred_w_c = dw_c * w + w_c
    pred_h = dh.exp() * h
    pred_w = dw.exp() * w

    pred_box = delta.clone()
    pred_box[:, :, 1::4] = pred_h_c - 0.5 * pred_h
    pred_box[:, :, 0::4] = pred_w_c - 0.5 * pred_w
    pred_box[:, :, 3::4] = pred_h_c + 0.5 * pred_h
    pred_box[:, :, 2::4] = pred_w_c + 0.5 * pred_w
    return pred_box


def box_clip(box, im_info):
    for i in range(box.size(0)):
        box[i, :, 1:4:2].clamp_(0, im_info[i, 0])
        box[i, :, 0:4:2].clamp_(0, im_info[i, 1])
    return box


def box_iou(box_a, box_b):
    def _func(ind):
        return torch.max if ind < 2 else torch.min

    w_0, h_0, w_1, h_1 = [_func(ind)(box_a[:, ind::4], box_b[:, ind::4].t()) for ind in range(4)] # h_0's shape is b,k
    inter_area = (h_1 - h_0).clamp(min=0) * (w_1 - w_0).clamp(min=0)  # b,k
    a_area = (box_a[:, 2::4] - box_a[:, 0::4]) * (box_a[:, 3::4] - box_a[:, 1::4])  # b,
    b_area = (box_b[:, 2::4] - box_b[:, 0::4]) * (box_b[:, 3::4] - box_b[:, 1::4])  # k,
    union_area = a_area + b_area.t() - inter_area
    iou = inter_area / union_area  # b,k
    return iou  # b,k


def box_loc(box_a, box_b):
    a_h, a_w, a_h_c, a_w_c = hw_c(box_a)
    b_h, b_w, b_h_c, b_w_c = hw_c(box_b)

    d_h_c = (b_h_c - a_h_c) / a_h
    d_w_c = (b_w_c - a_w_c) / a_w
    d_h = (b_h / a_h).log()
    d_w = (b_w / a_w).log()
    return torch.stack((d_w_c, d_h_c, d_w, d_h), -1)


def box_union(box_a, box_b):
    def _func(ind):
        return torch.min if ind < 2 else torch.max

    w_0, h_0, w_1, h_1 = [_func(ind)(box_a[:, ind::4], box_b[:, ind::4]) for ind in range(4)]
    return torch.cat((w_0, h_0, w_1, h_1), -1)


