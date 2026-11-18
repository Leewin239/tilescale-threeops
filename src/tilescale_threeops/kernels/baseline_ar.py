# Adapted from TileScale. See third_party/LICENSE.TileScale.
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit(
    compile_once=True,
)
def gemm_ar_sm_specialized_kernel(
    M,
    N,
    K,
    num_ranks,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    group_size_m: int,
    pipeline_stages: int,
    ar_block_e: int,
    dtype=T.bfloat16,
):
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K, block_K)
    total_tiles = m_blocks * n_blocks
    ar_rows = ar_block_e // block_N
    ar_row_chunks = block_M // ar_rows
    # store_block_M = block_M // 2
    accum_dtype = T.float32

    def tile_coords(tile_id):
        super_rows = (m_blocks // group_size_m) * group_size_m
        final_rows = m_blocks - super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks
        is_super_tile = tile_id < super_rows * n_blocks
        remainder_id = tile_id - super_rows * n_blocks
        by = T.if_then_else(
            is_super_tile,
            group_size_m * (tile_id // super_tiles) + tile_id % group_size_m,
            super_rows + remainder_id % final_rows_safe,
        )
        bx = T.if_then_else(
            is_super_tile,
            (tile_id % super_tiles) // group_size_m,
            remainder_id // final_rows_safe,
        )
        return by, bx

    @T.macro
    def materialized_tile_coords(tile_id):
        by_expr, bx_expr = tile_coords(tile_id)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C_local_out: T.Tensor((M, N), dtype),
        mcast_C: T.Tensor((M, N), dtype),
        mcast_signal: T.Tensor((total_tiles,), T.uint32),
        local_signal: T.Tensor((total_tiles,), T.uint32),
        num_comm_sms: T.int32,
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            num_comp_sms = sm_num - num_comm_sms
            waves = T.ceildiv(total_tiles, num_comp_sms)
            if bid < num_comp_sms:
                with T.sm_specialize_scope(auto_ws=True):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_K, block_N), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    for w in T.serial(waves):
                        tile_id = bid + w * num_comp_sms
                        if tile_id < total_tiles:
                            by, bx = materialized_tile_coords(tile_id)
                            T.clear(C_local)
                            for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                                T.copy(A[by * block_M, k * block_K], A_shared)
                                T.copy(B[k * block_K, bx * block_N], B_shared)
                                T.gemm(A_shared, B_shared, C_local)
                            T.copy(C_local, C_local_out[by * block_M, bx * block_N])
                            T.fence_sys()
                            if T.get_thread_binding(0) == 0:
                                T.multimem_signal_add(mcast_signal[tile_id], 1)
            else:
                with T.sm_specialize_scope(auto_ws=False):
                    comm_sm_id = bid - num_comp_sms
                    acc = T.alloc_fragment((ar_rows, block_N), dtype)
                    for task_iter in T.serial(T.ceildiv(total_tiles, num_comm_sms * num_ranks)):
                        tile_id = (task_iter * num_comm_sms + comm_sm_id) * num_ranks + local_rank
                        if tile_id < total_tiles:
                            by, bx = materialized_tile_coords(tile_id)
                            if T.get_thread_binding(0) == 0:
                                T.wait_ge(
                                    local_signal[tile_id],
                                    num_ranks,
                                    scope=T.WaitScope.SYS,
                                    semantics=T.WaitSemantics.ACQUIRE,
                                )
                            T.sync_threads()
                            for row_chunk in T.serial(ar_row_chunks):
                                row = row_chunk * ar_rows
                                T.multimem_ld_reduce(
                                    mcast_C[
                                        by * block_M + row : by * block_M + row + ar_rows,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                    acc,
                                    reduce_op=T.MultimemReduceOp.ADD,
                                )
                                T.multimem_st(
                                    acc,
                                    mcast_C[
                                        by * block_M + row : by * block_M + row + ar_rows,
                                        bx * block_N : (bx + 1) * block_N,
                                    ],
                                )

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def reset_gemm_ar_specialized_state_kernel(num_tiles, threads=256):
    @T.prim_func
    def main(local_signal: T.Tensor((num_tiles,), T.uint32)):
        with T.Kernel(T.ceildiv(num_tiles, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            idx = bx * threads + tid
            if idx < num_tiles:
                local_signal[idx] = 0

    return main

