# coding=utf-8
import cupy
import math
from functools import namedtuple


__all__ = ['load_kernel', 'get_grid_block', 'Stream']


@cupy.util.memoize(True)
def load_kernel(kernel_name, code, wrap_func=None):
    cupy.cuda.runtime.free(0)
    kernel_code = cupy.cuda.compile_with_cache(code)
    func = kernel_code.get_function(kernel_name)
    if wrap_func:
        return wrap_func(func)
    return func


THREAD_PER_BLOCK = 1024


def get_grid_block(num, threads_per_block=None):
    threads_per_block = threads_per_block or THREAD_PER_BLOCK
    block = (threads_per_block, 1, 1)
    grid_x = math.ceil(num / threads_per_block)
    grid = (grid_x, 1, 1)
    return grid, block


Stream = namedtuple('Stream', ['ptr'])


