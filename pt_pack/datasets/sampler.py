# coding=utf-8
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler
import torch
from itertools import accumulate, chain, repeat, tee


__all__ = ['FixBatchSampler']


class FixBatchSampler(Sampler):
    def __init__(self, dataset, num_workers, batch_size, shuffle, drop_last=False):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_workers = num_workers
        if shuffle:
            self.indices = torch.randperm(len(self.dataset)).tolist()
        else:
            self.indices = list(range(len(self.dataset)))
        self.worker_indices = self.chunk(self.indices, self.num_workers)
        self.samplers = [BatchSampler(SubsetRandomSampler(indices), batch_size, drop_last) for indices in self.worker_indices]

    def chunk(self, xs, n):
        L = len(xs)
        s, r = divmod(L, n)
        widths = chain(repeat(s + 1, r), repeat(s, n - r))
        offsets = accumulate(chain((0,), widths))
        b, e = tee(offsets)
        next(e)
        return [xs[s] for s in map(slice, b, e)]

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        return (next(iters[idx % self.num_workers]) for idx in range(len(self)))

    def __len__(self):
        return sum([len(sampler) for sampler in self.samplers])




