# coding=utf-8


def smooth_l1_loss(loc_pred, loc_label, loc_in_weight, loc_out_weight, sigma, dims):
    sigma2 = sigma ** 2
    abs_diff = (loc_in_weight * (loc_pred - loc_label)).abs()
    flag = (abs_diff < (1. / sigma2)).detach().float()
    loss_box = flag * (sigma2 / 2.) * (abs_diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2)
    loss_box = loss_box * loc_out_weight
    for dim in dims:
        loss_box = loss_box.sum(dim)
    return loss_box.mean()
