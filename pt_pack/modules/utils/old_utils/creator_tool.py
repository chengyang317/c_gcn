# coding=utf-8
import torch
import numpy as np
from .bbox_tools import base_anchor, box_update, box_clip, box_iou, box_loc
from modules.utils.old_utils.nms.non_maximum_suppression import non_maximum_suppression
from functools import reduce
import torch.nn.functional as F
from pt_pack.utils import totensor


__all__ = ['rand_perm', 'AnchorProposalCreator', 'AnchorTargetCreator', 'ProposalTargetCreator',
           'ProposalEvalCreator', 'AnchorCreateCreator']


def rand_perm(size, is_cuda=True):
    ind = torch.randperm(size)
    return ind if not is_cuda else ind.cuda()


class AnchorCreateCreator(object):
    """
    Create an anchor for feature shape (h, w)
    the returned anchor shape is (batch_size, feat_h, feat_w, 4*base_anchor_size)
    """
    def __init__(self, feat_stride=16., ratios=(0.5, 1, 2), scales=(8, 16, 32)):
        self.feat_stride = feat_stride
        self.base_anchor = base_anchor(ratios=ratios, scales=scales)  # a, 4

    def __call__(self, batch_size, feat_h, feat_w, is_cuda=True):
        # todo : use cuda to speed
        shift_w = np.arange(0, feat_w) * self.feat_stride
        shift_h = np.arange(0, feat_h) * self.feat_stride
        shift_w, shift_h = np.meshgrid(shift_w, shift_h)
        shifts = np.vstack((shift_w.ravel(), shift_h.ravel(), shift_w.ravel(), shift_h.ravel())).transpose()
        shifts = totensor(shifts, cuda=True).contiguous().float()  # k, 4
        base_anchor = self.base_anchor.cuda(shifts.get_device())
        anchor = base_anchor.unsqueeze(0) + shifts.unsqueeze(1)  # k, 12, 4
        anchor = anchor.view(feat_h, feat_w, -1)  # h, w, 48
        anchor = anchor.unsqueeze(0).expand(batch_size, feat_h, feat_w, -1).contiguous()
        return anchor


class AnchorProposalCreator(object):
    """
    Based on anchor and the predicted loc, we can get the finale box location prediction. The we sort the location
    prediction using the probability of objectness. Lastly, None maximum suppresion method is adopted to refine the
    location prediction.
    """
    def __init__(self, opt):
        self.opt = opt
        from pt_pack.modules.nms import NMS
        self.nms = NMS()

    def __call__(self, cls_prob, loc_pred, im_info, anchor, is_train):
        """

        :param cls_prob: cuda tensor data, can't be Variable, its shape is b, 2*12, h, w
        :param loc_pred: its shape is b, 4*12, h, w
        :param im_info: shape is b, 3. it contains origin image size h, w and scale
        :param anchor: shape is b, h, w, 48
        :param is_train:
        :return:
        """
        opt = self.opt['train' if is_train else 'test']
        keys = ['pre_nms_n', 'pos_nms_n', 'nms_thresh', 'min_size']
        pre_nms_n, post_nms_n, nms_thresh, min_size = [opt[key] for key in keys]

        batch_size, base_anchor_size, feat_h, feat_w = cls_prob.size()
        base_anchor_size = int(base_anchor_size / 2)

        anchor = anchor.view(batch_size, -1, 4)
        loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view_as(anchor)

        # Same story for the scores:
        pos_prob = cls_prob[:, base_anchor_size:, :, :]
        pos_prob = pos_prob.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)  # b, n

        # 1.Convert anchors into proposals using delta
        proposal = box_update(anchor, loc_pred)  # b, n, 4

        # 2. clip predicted boxes to image
        proposal = box_clip(proposal, im_info)

        # 3.
        _, sort_ind = pos_prob.sort(dim=1, descending=True)  # b, n
        if pre_nms_n > 0:
            sort_ind = sort_ind[:, :pre_nms_n]

        proposal = proposal.gather(dim=1, index=sort_ind.unsqueeze(-1).expand(-1, -1, 4))
        # pos_prob = pos_prob.gather(dim=1, index=sort_ind)

        reduced_proposal = proposal.new(batch_size, post_nms_n, 4).zero_()
        for b_ind in range(batch_size):
            proposal_s = proposal[b_ind]
            keep_idx = totensor(non_maximum_suppression(proposal_s, thresh=nms_thresh), cuda=anchor.is_cuda).view(-1).long()
            a = self.nms(proposal_s, nms_thresh)
            if post_nms_n > 0:
                keep_idx = keep_idx[:post_nms_n]
            reduced_proposal[b_ind, :keep_idx.numel(), :] = proposal_s[keep_idx]
        return reduced_proposal


class AnchorTargetCreator(object):
    def __init__(self, opt):
        self.target_size = opt['target_size']
        self.pos_iou_thresh = opt['pos_iou_thresh']
        self.neg_iou_thresh = opt['neg_iou_thresh']
        self.pos_ratio = opt['pos_ratio']
        self.allowed_border = opt['allowed_border']
        self.pos_anchor_size = int(self.pos_ratio * self.target_size)
        self.neg_anchor_size = self.target_size - self.pos_anchor_size

    def __call__(self, anchor, gt_label, im_info):
        """

        :param anchor: shape as (b, h, w, 48)
        :param gt_label: shape as (b, bbox_size, 5)
        :param im_info:
        :return:
        """
        anchor = anchor.view(anchor.size(0), -1, 4)
        batch_size, anchor_size, _ = anchor.size()
        is_cuda = anchor.is_cuda
        anchor_keep_ind = anchor.new(batch_size, self.target_size).long()
        anchor_cls_label = anchor.new(batch_size, self.target_size).zero_().long()
        anchor_loc_label = anchor.new(batch_size, self.target_size, 4).fill_(-1)

        for b_id in range(batch_size):
            gt_label_s = gt_label[b_id]
            gt_label_s = gt_label_s[(gt_label_s[:, 4] != -1).nonzero().view(-1)]
            gt_loc_label_s = gt_label_s[:, :4]
            anchor_s = anchor[b_id]
            im_h, im_w = im_info[b_id, 0], im_info[b_id, 1]
            ind_s = ((anchor_s[:, 0] >= -self.allowed_border)
                     & (anchor_s[:, 1] >= -self.allowed_border)
                     & (anchor_s[:, 2] <= im_w+self.allowed_border)
                     & (anchor_s[:, 3] <= im_h+self.allowed_border)).nonzero().view(-1)
            anchor_s = anchor_s[ind_s]
            label_s = gt_label.new(ind_s.size(0)).fill_(-1).long()
            iou_s = box_iou(anchor_s, gt_loc_label_s)  # n, k
            max_iou_s, argmax_iou_s = iou_s.max(-1)
            gt_max_ious, gt_argmax_iou_s = iou_s.max(0)
            label_s[max_iou_s < self.neg_iou_thresh] = 0
            # label_s[gt_argmax_iou_s] = 1
            is_max = iou_s.eq(gt_max_ious.expand_as(iou_s)).sum(-1)
            label_s[is_max > 0] = 1
            label_s[max_iou_s >= self.pos_iou_thresh] = 1
            label_pos_ind = (label_s == 1).nonzero().view(-1)
            label_neg_ind = (label_s == 0).nonzero().view(-1)
            label_pos_ind_size = label_pos_ind.numel()
            label_neg_ind_size = label_neg_ind.numel()
            if label_pos_ind_size < self.pos_anchor_size:
                neg_anchor_size = self.target_size - label_pos_ind_size
                neg_anchor_size = min(label_neg_ind_size, neg_anchor_size)
                label_neg_ind = label_neg_ind[rand_perm(label_neg_ind_size, is_cuda)[:neg_anchor_size]]
            elif label_neg_ind_size < self.neg_anchor_size:
                pos_anchor_size = self.target_size - label_neg_ind_size
                pos_anchor_size = min(label_pos_ind_size, pos_anchor_size)
                label_pos_ind = label_pos_ind[rand_perm(label_pos_ind_size, is_cuda)[:pos_anchor_size]]
            else:
                label_pos_ind = label_pos_ind[rand_perm(label_pos_ind_size, is_cuda)[:self.pos_anchor_size]]
                label_neg_ind = label_neg_ind[rand_perm(label_neg_ind_size, is_cuda)[:self.neg_anchor_size]]
            label_keep_ind = torch.cat((label_pos_ind, label_neg_ind))

            anchor_keep_ind[b_id] = ind_s[label_keep_ind]
            anchor_cls_label[b_id, :label_pos_ind.numel()] = 1
            anchor_loc_label[b_id, :label_pos_ind.numel()] = box_loc(anchor_s[label_pos_ind],
                                                                     gt_loc_label_s[argmax_iou_s[label_pos_ind]])
        return anchor_keep_ind, anchor_cls_label, anchor_loc_label


class ProposalTargetCreator(object):
    def __init__(self, opt):
        self.target_size = opt['target_size']
        self.pos_iou_thresh = opt['pos_iou_thresh']
        self.neg_iou_thresh_hi = opt['neg_iou_thresh_hi']
        self.neg_iou_thresh_lo = opt['neg_iou_thresh_lo']  # NOTE: py-faster-rcnn默认的值是0.1
        self.box_loc_norm_mean = totensor(np.asarray(opt['box_loc_norm_mean'], dtype=np.float32))
        self.box_loc_norm_std = totensor(np.asarray(opt['box_loc_norm_std'], dtype=np.float32))
        self.box_loc_norm_precomputed = opt['box_loc_norm_precomputed']
        self.pos_ratio = opt['pos_ratio']
        self.pos_proposal_size = int(self.pos_ratio * self.target_size)
        self.neg_proposal_size = self.target_size - self.pos_proposal_size

    def __call__(self, proposal, gt_label, is_train=True):
        """

        :param proposal: b, k, 4
        :param gt_label:b, a, 5,
        :return:
        """
        if is_train:
            proposal = torch.cat((proposal, gt_label[:, :, :-1]), dim=1)  # b, m+n, 4
        batch_size, _, label_dim = gt_label.size()
        proposal_keep_ind = proposal.new(batch_size, self.target_size).long()
        proposal_cls_label = proposal.new(batch_size, self.target_size, label_dim-4).zero_().long()
        proposal_loc_label = proposal.new(batch_size, self.target_size, 4).fill_(-1)
        proposal_argmax = proposal.new(batch_size, self.target_size).long()
        is_cuda = proposal.is_cuda

        for b_id in range(batch_size):
            gt_label_s = gt_label[b_id]
            gt_label_s = gt_label_s[(gt_label_s[:, 4] != -1).nonzero().view(-1)]
            proposal_s = proposal[b_id]
            # remove the padding gt_proposal contained within proposal
            valid_mask = reduce(lambda x, y: x | y, [proposal_s[:, ind] != 0. for ind in range(4)])
            iou_s = box_iou(proposal_s, gt_label_s[:, :4])  # n, k
            max_iou_s, argmax_iou_s = iou_s.max(-1)
            pos_ind = (max_iou_s >= self.pos_iou_thresh).nonzero().view(-1)
            neg_ind = ((max_iou_s < self.neg_iou_thresh_hi) & (max_iou_s >= self.neg_iou_thresh_lo) & valid_mask).nonzero().view(-1)
            pos_ind_size = pos_ind.numel()
            neg_ind_size = neg_ind.numel()
            if pos_ind_size < self.pos_proposal_size:
                neg_proposal_size = self.target_size - pos_ind_size
                neg_proposal_size = min(neg_ind_size, neg_proposal_size)
                neg_ind = neg_ind[rand_perm(neg_ind_size, is_cuda)[:neg_proposal_size]]
            elif neg_ind_size < self.neg_proposal_size:
                pos_proposal_size = self.target_size - neg_ind_size
                pos_proposal_size = min(pos_ind_size, pos_proposal_size)
                pos_ind = pos_ind[rand_perm(pos_ind_size, is_cuda)[:pos_proposal_size]]
            else:
                pos_ind = pos_ind[rand_perm(pos_ind_size, is_cuda)[:self.pos_proposal_size]]
                neg_ind = neg_ind[rand_perm(neg_ind_size, is_cuda)[:self.neg_proposal_size]]
            keep_ind = torch.cat((pos_ind, neg_ind))
            assert keep_ind.numel() == self.target_size
            proposal_keep_ind[b_id] = keep_ind
            proposal_argmax[b_id] = argmax_iou_s[keep_ind]
            proposal_cls_label[b_id][:pos_ind.numel()] = gt_label_s[:, 4:][argmax_iou_s[pos_ind]]
            proposal_loc = box_loc(proposal_s[pos_ind], gt_label_s[:, :4][argmax_iou_s[pos_ind]])
            if self.box_loc_norm_precomputed:
                proposal_loc = (proposal_loc - totensor(self.box_loc_norm_mean, is_cuda).expand_as(proposal_loc)) \
                               / totensor(self.box_loc_norm_std, is_cuda)
            proposal_loc_label[b_id][:pos_ind.numel()] = proposal_loc
        proposal = proposal.gather(dim=1, index=proposal_keep_ind.unsqueeze(-1).expand(-1, -1, 4))
        return proposal, proposal_keep_ind, proposal_cls_label, proposal_loc_label, proposal_argmax


class ProposalEvalCreator(object):
    def __init__(self, opt, class_num):
        self.opt = opt
        self.class_num = class_num

    def __call__(self, proposal, cls_score, loc_pred, im_info):
        is_cuda = proposal.is_cuda
        batch_num, proposal_num, _ = proposal.shape
        cls_prob = F.softmax(cls_score, dim=-1).data  # b, 256, n_class
        if self.opt['box_loc_norm_precomputed']:
            norm_std = totensor(self.opt['box_loc_norm_std'], is_cuda)
            norm_mean = totensor(self.opt['box_loc_norm_mean'], is_cuda)
            loc_pred = (loc_pred.view(batch_num, -1, 4) * norm_std + norm_mean).view(batch_num, proposal_num, -1)
        box_pred = box_clip(box_update(proposal, loc_pred), im_info)
        box_pred = box_pred / im_info[:, 2].unsqueeze(1).expand_as(box_pred)  # b, proposal_size, 4*num_class
        total_box = []
        for batch_id in range(batch_num):
            box_per_batch = []
            for class_id in range(1, self.class_num):
                prob_per_class = cls_prob[batch_id, :, class_id].view(-1)  # 300
                pos_ind = prob_per_class.ge(self.opt['prob_thresh']).nonzero().view(-1)
                if not pos_ind.numel():
                    continue
                prob_per_class = prob_per_class[pos_ind]
                box_per_class = box_pred[batch_id, pos_ind, class_id * 4:(class_id + 1) * 4]  # 300, 4
                _, sort_ind = prob_per_class.sort(dim=0, descending=True)
                box_per_class = box_per_class[sort_ind]
                prob_per_class = prob_per_class[sort_ind]
                keep = totensor(non_maximum_suppression(box_per_class, thresh=self.opt['nms_thresh'])).view(-1).long()
                box_per_class = torch.cat(
                    [box_per_class[keep], prob_per_class.new(keep.numel(), 1).fill_(class_id),
                     prob_per_class.new(keep.numel(), 1).fill_(im_info[batch_id, -1].item()),
                     prob_per_class[keep].unsqueeze(-1)]
                    , dim=1)
                box_per_batch.append(box_per_class)
            box_per_batch = torch.cat(box_per_batch, dim=0)
            if self.opt['box_num_per_img'] < len(box_per_batch):
                _, keep_ind = box_per_batch[:, -1].sort(dim=0, descending=True)
                keep_ind = keep_ind[:self.opt['box_num_per_img']]
                box_per_batch = box_per_batch[keep_ind]
            total_box.append(box_per_batch)
        total_box = torch.cat(total_box, dim=0)
        return total_box











