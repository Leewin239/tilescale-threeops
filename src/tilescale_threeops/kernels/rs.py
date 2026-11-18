# Adapted from TileScale. See third_party/LICENSE.TileScale.
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit(
    compile_once=True,
)
def gemm_rs_specialized_kernel(
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
    use_tma_epilogue: bool,
    dtype=T.bfloat16,
    grid_ctas: int = 80,
    weight_nk: bool = False,
    single_tile_epilogue: bool = False,
    local_first: bool = False,
    paired_store: bool = False,
):
    sm_num = grid_ctas
    K_per_rank = K // num_ranks
    weight_shape = (N, K_per_rank) if weight_nk else (K_per_rank, N)
    weight_tile = (block_N, block_K) if weight_nk else (block_K, block_N)
    M_per_rank = M // num_ranks
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    k_blocks = T.ceildiv(K_per_rank, block_K)
    blocks_per_rank = m_blocks // num_ranks
    total_tiles = m_blocks * n_blocks
    waves = T.ceildiv(total_tiles, sm_num)
    accum_dtype = T.float32

    store_block_M = block_M if single_tile_epilogue else block_M // 2
    if single_tile_epilogue and not use_tma_epilogue:
        raise ValueError("single-tile RS epilogue requires TMA store-add")
    if paired_store and (single_tile_epilogue or not use_tma_epilogue):
        raise ValueError("paired RS stores require the two-tile TMA epilogue")

    def tile_coords(tile_id, local_rank):
        rotation = ((local_rank + (0 if local_first else 1)) * blocks_per_rank * n_blocks) % total_tiles
        rotated = (tile_id + rotation) % total_tiles
        super_rows = (m_blocks // group_size_m) * group_size_m
        final_rows = m_blocks - super_rows
        final_rows_safe = T.max(final_rows, 1)
        super_tiles = group_size_m * n_blocks
        is_super_tile = rotated < super_rows * n_blocks
        remainder_id = rotated - super_rows * n_blocks

        by = T.if_then_else(
            is_super_tile,
            group_size_m * (rotated // super_tiles) + rotated % group_size_m,
            super_rows + remainder_id % final_rows_safe,
        )
        bx = T.if_then_else(
            is_super_tile,
            (rotated % super_tiles) // group_size_m,
            remainder_id // final_rows_safe,
        )
        return by, bx

    @T.macro
    def materialized_tile_coords(tile_id, local_rank):
        by_expr, bx_expr = tile_coords(tile_id, local_rank)
        by = T.alloc_var(T.int32)
        bx = T.alloc_var(T.int32)
        by = by_expr
        bx = bx_expr
        return by, bx

    if single_tile_epilogue:
        @T.prim_func
        def single_tile(
            A: T.Tensor((M, K_per_rank), dtype),
            B: T.Tensor(weight_shape, dtype),
            C: T.Tensor((M_per_rank, N), dtype),
        ):
            with T.Kernel(sm_num, threads=threads) as bid:
                local_rank = T.get_rank()
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared(weight_tile, dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                for w in T.serial(waves):
                    tile_id = bid + w * sm_num
                    if tile_id < total_tiles:
                        by, bx = materialized_tile_coords(tile_id, local_rank)
                        dst_rank = by // blocks_per_rank
                        local_by = by - dst_rank * blocks_per_rank
                        T.clear(C_local)
                        for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                            T.copy(A[by * block_M, k * block_K], A_shared)
                            if weight_nk:
                                T.copy(B[bx * block_N, k * block_K], B_shared)
                            else:
                                T.copy(B[k * block_K, bx * block_N], B_shared)
                            T.gemm(A_shared, B_shared, C_local, transpose_B=weight_nk)
                        T.copy(C_local, C_shared)
                        T.sync_threads(0, threads)
                        T.atomic_add(
                            C[local_by * block_M : (local_by + 1) * block_M,
                              bx * block_N : (bx + 1) * block_N],
                            C_shared, use_tma=True, tma_wait_complete=True, dst_pe=dst_rank,
                        )
                        T.sync_threads(0, threads)
        return single_tile

    @T.prim_func
    def main(
        A: T.Tensor((M, K_per_rank), dtype),
        B: T.Tensor(weight_shape, dtype),
        C: T.Tensor((M_per_rank, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as bid:
            local_rank = T.get_rank()
            A_shared_low = T.alloc_shared((store_block_M, block_K), dtype)
            A_shared_high = T.alloc_shared((store_block_M, block_K), dtype)
            B_shared = T.alloc_shared(weight_tile, dtype)
            C_local_low = T.alloc_fragment((store_block_M, block_N), accum_dtype)
            C_local_high = T.alloc_fragment((store_block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((store_block_M, block_N), dtype)
            if paired_store:
                C_shared_high = T.alloc_shared((store_block_M, block_N), dtype)

            for w in T.serial(waves):
                tile_id = bid + w * sm_num
                if tile_id < total_tiles:
                    by, bx = materialized_tile_coords(tile_id, local_rank)
                    dst_rank = by // blocks_per_rank
                    local_by = by - dst_rank * blocks_per_rank

                    T.clear(C_local_low)
                    T.clear(C_local_high)
                    for k in T.Pipelined(k_blocks, num_stages=pipeline_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared_low)
                        T.copy(A[by * block_M + store_block_M, k * block_K], A_shared_high)
                        if weight_nk:
                            T.copy(B[bx * block_N, k * block_K], B_shared)
                        else:
                            T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared_low, B_shared, C_local_low, transpose_B=weight_nk)
                        T.gemm(A_shared_high, B_shared, C_local_high, transpose_B=weight_nk)

                    # Fused reduce-scatter
                    T.copy(C_local_low, C_shared)
                    T.sync_threads(0, threads)
                    if use_tma_epilogue:
                        # TMA store-add into peer C.
                        T.atomic_add(
                            C[
                                local_by * block_M : local_by * block_M + store_block_M,
                                bx * block_N : (bx + 1) * block_N,
                            ],
                            C_shared,
                            use_tma=True,
                            tma_wait=not paired_store,
                            tma_wait_complete=not paired_store,
                            dst_pe=dst_rank,
                        )
                    else:
                        # vectorized bf16x2 red.global.add.
                        tid = T.get_thread_binding()
                        for idx in T.serial(tid, store_block_M * block_N // 2, threads):
                            elem = idx * 2
                            T.atomic_addx2(
                                C[local_by * block_M + elem // block_N, bx * block_N + elem % block_N],
                                C_shared[elem // block_N, elem % block_N],
                                dst_pe=dst_rank,
                            )
                    T.sync_threads(0, threads)

                    if paired_store:
                        T.copy(C_local_high, C_shared_high)
                    else:
                        T.copy(C_local_high, C_shared)
                    T.sync_threads(0, threads)
                    if use_tma_epilogue:
                        # TMA store-add into peer C.
                        T.atomic_add(
                            C[
                                local_by * block_M + store_block_M : local_by * block_M + block_M,
                                bx * block_N : (bx + 1) * block_N,
                            ],
                            C_shared_high if paired_store else C_shared,
                            use_tma=True,
                            tma_wait_complete=True,
                            dst_pe=dst_rank,
                        )
                    else:
                        # vectorized bf16x2 red.global.add.
                        tid = T.get_thread_binding()
                        for idx in T.serial(tid, store_block_M * block_N // 2, threads):
                            elem = idx * 2
                            T.atomic_addx2(
                                C[local_by * block_M + store_block_M + elem // block_N, bx * block_N + elem % block_N],
                                C_shared[elem // block_N, elem % block_N],
                                dst_pe=dst_rank,
                            )
                    T.sync_threads(0, threads)

    return main

