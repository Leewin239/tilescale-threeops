# Adapted from TileScale. See third_party/LICENSE.TileScale.
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_once=True,
)
def ag_gemm_sm_specialized_kernel(
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
    use_tma_store: bool,
    dtype=T.bfloat16,
    grid_ctas: int = 80,
    weight_nk: bool = False,
    defer_tma_writeback: bool = False,
):
    if defer_tma_writeback and not use_tma_store:
        raise ValueError("deferred writeback requires the TMA epilogue")
    sm_num = grid_ctas
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    weight_shape = (N_per_rank, K) if weight_nk else (K, N_per_rank)
    weight_tile = (block_N, block_K) if weight_nk else (block_K, block_N)
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N_per_rank, block_N)
    local_m_blocks = T.ceildiv(M_per_rank, block_M)
    k_blocks = T.ceildiv(K, block_K)
    comm_block_M = block_M * 2
    comm_block_K = block_K * 2
    store_block_M = block_M if block_N <= 128 else block_M // 2
    comm_chunks = 2
    local_comm_m_blocks = T.ceildiv(M_per_rank, comm_block_M)
    comm_m_blocks = T.ceildiv(M, comm_block_M)
    comm_k_blocks = T.ceildiv(K, comm_block_K)
    comm_tasks_per_rank = local_comm_m_blocks * comm_k_blocks
    total_tiles = m_blocks * n_blocks
    accum_dtype = T.float32
    local_tiles = local_m_blocks * n_blocks

    def tile_coords(tile_id, local_rank):
        """Local-first tile scheduler.

        Local tiles use Super-M grouping for B-tile reuse. Remote tiles then
        walk peer ranks round-robin inside each remote row shard.
        """
        is_local = tile_id < local_tiles
        local_super_rows = (local_m_blocks // group_size_m) * group_size_m
        final_rows = local_m_blocks - local_super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks

        is_super_tile = tile_id < local_super_rows * n_blocks
        local_remainder_id = tile_id - local_super_rows * n_blocks
        local_by = T.if_then_else(
            is_super_tile,
            group_size_m * (tile_id // super_tiles) + tile_id % group_size_m,
            local_super_rows + local_remainder_id % final_rows_safe,
        )
        local_bx = T.if_then_else(
            is_super_tile,
            (tile_id % super_tiles) // group_size_m,
            local_remainder_id // final_rows_safe,
        )

        remote_tile_id = tile_id - local_tiles
        target_shard = remote_tile_id // ((num_ranks - 1) * n_blocks)
        idx_in_shard = remote_tile_id % ((num_ranks - 1) * n_blocks)
        peer_rank_offset = idx_in_shard % (num_ranks - 1)
        peer_rank = peer_rank_offset + T.if_then_else(peer_rank_offset >= local_rank, 1, 0)
        remote_by = peer_rank * local_m_blocks + target_shard
        remote_bx = idx_in_shard // (num_ranks - 1)

        by = T.if_then_else(is_local, local_rank * local_m_blocks + local_by, remote_by)
        bx = T.if_then_else(is_local, local_bx, remote_bx)
        return is_local, by, bx, by - local_rank * local_m_blocks

    @T.macro
    def materialized_tile_coords(tile_id, local_rank):
        _, by_expr, bx_expr, local_by_expr = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        local_by = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        local_by = local_by_expr
        return by, bx, local_by

    @T.macro
    def materialized_global_tile_coords(tile_id, local_rank):
        _, by_expr, bx_expr, _ = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx

    @T.prim_func
    def main(
        A_local: T.Tensor((M_per_rank, K), dtype),
        B: T.Tensor(weight_shape, dtype),
        mcast_A: T.Tensor((M, K), dtype),
        gathered_A: T.Tensor((M, K), dtype),
        mcast_signal: T.Tensor((comm_m_blocks,), T.uint32),
        local_signal: T.Tensor((comm_m_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        C: T.Tensor((M, N_per_rank), dtype),
        num_comm_sms: T.int32,
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            num_comp_sms = sm_num - num_comm_sms
            waves = T.ceildiv(total_tiles, num_comp_sms)
            comm_workers_per_signal = T.min(num_comm_sms * comm_chunks, comm_k_blocks)
            T.barrier_blocks(barriers[0, 0])
            if bid < num_comp_sms:
                with T.sm_specialize_scope(auto_ws=True):
                    A_comp_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_comp_shared = T.alloc_shared(weight_tile, dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    if use_tma_store:
                        # A 128x128 epilogue fits in SMEM; wider 128x256 tiles
                        # are staged as two 64x256 TMA stores to stay within
                        # Hopper SMEM budget after auto warp-specialized input
                        # staging.
                        C_shared = T.alloc_shared((store_block_M, block_N), dtype)
                        C_local_cast = T.alloc_fragment((block_M, block_N), dtype)
                    for w in T.serial(waves):
                        tile_id = bid + w * num_comp_sms
                        if tile_id < total_tiles:
                            is_local_by, by_expr, _, _ = tile_coords(tile_id, local_rank)
                            load_by, load_bx, load_local_by = materialized_tile_coords(tile_id, local_rank)
                            if not is_local_by:
                                T.wait_ge(local_signal[by_expr // 2], comm_workers_per_signal)

                            T.clear(C_local)
                            for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                                if is_local_by:
                                    T.copy(A_local[load_local_by * block_M, k * block_K], A_comp_shared)
                                else:
                                    T.copy(gathered_A[load_by * block_M, k * block_K], A_comp_shared)
                                if weight_nk:
                                    T.copy(B[load_bx * block_N, k * block_K], B_comp_shared)
                                else:
                                    T.copy(B[k * block_K, load_bx * block_N], B_comp_shared)
                                T.gemm(A_comp_shared, B_comp_shared, C_local, transpose_B=weight_nk)
                            store_by, store_bx = materialized_global_tile_coords(tile_id, local_rank)
                            if use_tma_store:
                                T.copy(C_local, C_local_cast)
                                for row_offset in T.serial(0, block_M, store_block_M):
                                    if defer_tma_writeback:
                                        # The preceding store may overlap the next GEMM, but
                                        # must finish reading shared memory before it is reused.
                                        T.tma_store_wait(0)
                                        T.sync_threads(0, threads)
                                    for i, j in T.Parallel(store_block_M, block_N):
                                        C_shared[i, j] = C_local_cast[row_offset + i, j]
                                    if defer_tma_writeback:
                                        T.fence_proxy_async()
                                    T.sync_threads(0, threads)
                                    if defer_tma_writeback:
                                        T.tma_copy(C_shared, C[
                                            store_by * block_M + row_offset : store_by * block_M + row_offset + store_block_M,
                                            store_bx * block_N : (store_bx + 1) * block_N,
                                        ], leader_thread_extent=threads)
                                    else:
                                        T.copy(C_shared, C[
                                            store_by * block_M + row_offset : store_by * block_M + row_offset + store_block_M,
                                            store_bx * block_N : (store_bx + 1) * block_N,
                                        ])
                                        T.sync_threads(0, threads)
                            else:
                                T.copy(
                                    C_local,
                                    C[
                                        store_by * block_M,
                                        store_bx * block_N,
                                    ],
                                )
                    if defer_tma_writeback:
                        T.tma_store_wait(0, False)
            else:
                with T.sm_specialize_scope(auto_ws=False):
                    A_comm_shared = T.alloc_shared((comm_chunks, comm_block_M, comm_block_K), dtype)
                    A_comm_ready = T.alloc_barrier([1] * comm_chunks)
                    comm_sm_id = bid - num_comp_sms
                    warp_id = T.get_warp_idx_sync()
                    lane_id = T.get_lane_idx()
                    comm_chunk = warp_id % comm_chunks
                    if warp_id < comm_chunks:
                        # Each participating warp owns one comm chunk: it issues
                        # an independent TMA load/store pair into its own shared
                        # buffer and mbarrier.
                        for local_task_iter in T.serial(T.ceildiv(comm_tasks_per_rank, num_comm_sms * comm_chunks)):
                            local_task_id = (local_task_iter * num_comm_sms + comm_sm_id) * comm_chunks + comm_chunk
                            if local_task_id < comm_tasks_per_rank:
                                local_comm_by = local_task_id // comm_k_blocks
                                k = local_task_id - local_comm_by * comm_k_blocks
                                global_comm_by = local_rank * local_comm_m_blocks + local_comm_by
                                local_m_start = local_comm_by * comm_block_M
                                global_m_start = global_comm_by * comm_block_M
                                T.tma_copy(
                                    A_local[
                                        local_m_start : local_m_start + comm_block_M,
                                        k * comm_block_K : (k + 1) * comm_block_K,
                                    ],
                                    A_comm_shared[comm_chunk, :, :],
                                    barrier=A_comm_ready[comm_chunk],
                                    leader_thread_extent=32,
                                )
                                if lane_id == 0:
                                    T.barrier_arrive(A_comm_ready[comm_chunk])
                                T.mbarrier_wait_parity(A_comm_ready[comm_chunk], local_task_iter & 1)
                                T.tma_copy(
                                    A_comm_shared[comm_chunk, :, :],
                                    mcast_A[
                                        global_m_start : global_m_start + comm_block_M,
                                        k * comm_block_K : (k + 1) * comm_block_K,
                                    ],
                                    leader_thread_extent=32,
                                )
                                T.tma_store_wait(0, False)
                                if k + num_comm_sms * comm_chunks >= comm_k_blocks and lane_id == 0:
                                    T.multimem_signal_add(mcast_signal[global_comm_by], 1)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def reset_ag_gemm_specialized_state_kernel(signal_blocks, num_ranks, threads=256):
    @T.prim_func
    def main(
        local_signal: T.Tensor((signal_blocks,), T.uint32),
        barriers: T.Tensor((2, num_ranks), T.int32),
        do_barrier: T.bool,
    ):
        with T.Kernel(1, threads=threads):
            tid = T.get_thread_binding(0)
            if do_barrier:
                T.sync_blocks(barriers[1, 0])
            for i in T.serial(T.ceildiv(signal_blocks, threads)):
                signal_idx = i * threads + tid
                if signal_idx < signal_blocks:
                    local_signal[signal_idx] = 0
            if tid < num_ranks:
                barriers[0, tid] = 0
            if do_barrier:
                T.fence_sys()
                T.barrier_blocks(barriers[1, 0])

    return main

