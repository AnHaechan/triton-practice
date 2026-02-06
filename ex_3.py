import os
os.environ['TRITON_INTERPRET'] = '1'

import torch
import triton
import triton.language as tl

# moved util functions into separate file for better readability
from triton_util import cdiv, breakpoint_if, print_if

# m axis: matrix dim 1, block dim 0, offs_0, stride_0, max_0

@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1): 
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)

@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    # chunks along m/n/k dimensions
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)
    rk = get_1d_offset(size=bk, n_prev_chunks=0)
    # relevant offsets of a, b
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)
    # initialize and iteratively update accumulator
    acc = tl.zeros((bm, bn), dtype=tl.float32)

    # print(k, type(k))
    # print(bk, type(bk))
    
    for _ in range(0, k, bk):
        mask_a = get_2d_mask(rm, rk, m, k)
        mask_b = get_2d_mask(rk, rn, k, n)
        a = tl.load(offs_a, mask=mask_a, other=0.0)
        b = tl.load(offs_b, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False) # matmul in block ; Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        # increase offsets, so next iteration loads next chunks
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
        rk += bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)
    mask = get_2d_mask(rm, rn, m, n)
    tl.store(c, acc, mask=mask)

from functools import partial

def matmul(a, b, matmul_k_fn, bs=16, group_sz=None):
    assert a.shape[1] == b.shape[0], "matrix dims not compatible for matmul"
    # check_tensors_gpu_ready(a, b)
    (m, k), (_, n) = a.shape, b.shape
    # print(m,k,n)
    # print(type(k))
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
    group_sz = {} if group_sz is None else {"group_sz":group_sz} # not used in naive_matmul, but will be in grouped_matmul further below 
    matmul_k_fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1), # m k
        b.stride(0), b.stride(1), # k n
        c.stride(0), c.stride(1), # m n
        bm=bs, bn=bs, bk=bs, # Weirdness: allow_tf32 must be set to False for older GPUs, otherwise won't compile
        **group_sz
    )
    return c

naive_matmul = partial(matmul, matmul_k_fn=naive_matmul_k)

a = torch.ones((3, 4), dtype=torch.float32, device='cpu')
b = torch.ones((4, 5), dtype=torch.float32, device='cpu')

print(a, b)
print(naive_matmul(a, b))