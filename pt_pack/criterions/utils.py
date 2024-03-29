# coding=utf-8
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if target.dim() == 2:  # multians option
        _, target = torch.max(target, 1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res[f'acc@{k}'] = correct_k.mul_(100.0 / batch_size).item()
    return res