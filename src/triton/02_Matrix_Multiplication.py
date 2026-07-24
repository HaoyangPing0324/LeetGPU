import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    PID_M = tl.program_id(0)
    PID_K = tl.program_id(1)
    MAX_N = tl.cdiv(N, BLOCK_SIZE_N)

    accumulated_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    start_a_m = PID_M * BLOCK_SIZE_M
    a_m_offset = start_a_m + tl.arange(0, BLOCK_SIZE_M)

    start_b_k = PID_K * BLOCK_SIZE_K
    b_k_offset = start_b_k + tl.arange(0, BLOCK_SIZE_K)

    for n in tl.range(MAX_N):
        start_a_n = n * BLOCK_SIZE_N
        a_n_offset = start_a_n + tl.arange(0, BLOCK_SIZE_N)

        a_mn_mask = (a_m_offset[:, None] < M) & (a_n_offset[None, :] < N)
        a_mn_ptrs = a + a_m_offset[:, None] * stride_am + a_n_offset[None, :] * stride_an
        block_a_mn = tl.load(a_mn_ptrs, mask=a_mn_mask, other=0.0)

        start_b_n = n * BLOCK_SIZE_N
        b_n_offset = start_b_n + tl.arange(0, BLOCK_SIZE_N)


        b_nk_mask = (b_n_offset[:, None] < N) & (b_k_offset[None, :] < K)
        b_nk_ptrs = b + b_n_offset[:, None] * stride_bn + b_k_offset[None, :] * stride_bk

        block_b_nk = tl.load(b_nk_ptrs, mask=b_nk_mask, other=0.0)

        accumulated_block = tl.dot(block_a_mn, block_b_nk, accumulated_block, allow_tf32=False)
        # block_ab = tl.dot(block_a_mn, block_b_nk)
        # accumulated_block += block_ab

    c_mk_ptrs = c + a_m_offset[:, None] * stride_cm + b_k_offset[None, :] * stride_ck
    c_mk_mask = (a_m_offset[:, None] < M) & (b_k_offset[None, :] < K)
    tl.store(c_mk_ptrs, accumulated_block, mask=c_mk_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
    )
