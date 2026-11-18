# Adapted from TileScale. See third_party/LICENSE.TileScale.
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit(compile_once=True)
def ar_residual_rmsnorm_kernel(rows, hidden, eps=1e-6, dtype=T.bfloat16):
    @T.prim_func
    def main(
        AR: T.Tensor((rows, hidden), dtype),
        X: T.Tensor((rows, hidden), dtype),
        Weight: T.Tensor((hidden,), dtype),
        Residual: T.Tensor((rows, hidden), dtype),
        Normed: T.Tensor((rows, hidden), dtype),
    ):
        with T.Kernel(rows, threads=256) as row:
            ar_s = T.alloc_shared((1, hidden), dtype)
            x_s = T.alloc_shared((1, hidden), dtype)
            w_s = T.alloc_shared((hidden,), dtype)
            r_s = T.alloc_shared((1, hidden), dtype)
            n_s = T.alloc_shared((1, hidden), dtype)
            r_f = T.alloc_fragment((1, hidden), T.float32)
            sq = T.alloc_fragment((1, hidden), T.float32)
            sumsq = T.alloc_fragment((1,), T.float32)
            inv = T.alloc_fragment((1,), T.float32)
            T.copy(AR[row, 0], ar_s)
            T.copy(X[row, 0], x_s)
            T.copy(Weight, w_s)
            for j in T.Parallel(hidden):
                rounded = T.Cast(dtype, T.Cast(T.float32, ar_s[0, j]) +
                                 T.Cast(T.float32, x_s[0, j]))
                r_s[0, j] = rounded
                r_f[0, j] = T.Cast(T.float32, rounded)
                sq[0, j] = r_f[0, j] * r_f[0, j]
            T.reduce_sum(sq, sumsq, dim=1)
            inv[0] = T.rsqrt(sumsq[0] / T.float32(hidden) + T.float32(eps))
            for j in T.Parallel(hidden):
                n_s[0, j] = T.Cast(dtype, r_f[0, j] * inv[0] * T.Cast(T.float32, w_s[j]))
            T.copy(r_s, Residual[row, 0])
            T.copy(n_s, Normed[row, 0])

    return main


@tilelang.jit(compile_once=True)
def ar_residual_rmsnorm_backward_kernel(rows, hidden, eps=1e-6,
                                         dtype=T.bfloat16):
    @T.prim_func
    def main(
        Residual: T.Tensor((rows, hidden), dtype),
        Weight: T.Tensor((hidden,), dtype),
        DNorm: T.Tensor((rows, hidden), dtype),
        DResidual: T.Tensor((rows, hidden), dtype),
        DR: T.Tensor((rows, hidden), dtype),
        DWPartial: T.Tensor((rows, hidden), T.float32),
    ):
        with T.Kernel(rows, threads=256) as row:
            r_s = T.alloc_shared((1, hidden), dtype)
            w_s = T.alloc_shared((hidden,), dtype)
            dn_s = T.alloc_shared((1, hidden), dtype)
            dr_s = T.alloc_shared((1, hidden), dtype)
            out_s = T.alloc_shared((1, hidden), dtype)
            dw_s = T.alloc_shared((1, hidden), T.float32)
            r_f = T.alloc_fragment((1, hidden), T.float32)
            gw_f = T.alloc_fragment((1, hidden), T.float32)
            sq = T.alloc_fragment((1, hidden), T.float32)
            gr = T.alloc_fragment((1, hidden), T.float32)
            sumsq = T.alloc_fragment((1,), T.float32)
            sumgr = T.alloc_fragment((1,), T.float32)
            inv = T.alloc_fragment((1,), T.float32)
            T.copy(Residual[row, 0], r_s)
            T.copy(Weight, w_s)
            T.copy(DNorm[row, 0], dn_s)
            T.copy(DResidual[row, 0], dr_s)
            for j in T.Parallel(hidden):
                r_f[0, j] = T.Cast(T.float32, r_s[0, j])
                gw_f[0, j] = T.Cast(T.float32, dn_s[0, j]) * T.Cast(T.float32, w_s[j])
                sq[0, j] = r_f[0, j] * r_f[0, j]
                gr[0, j] = gw_f[0, j] * r_f[0, j]
            T.reduce_sum(sq, sumsq, dim=1)
            T.reduce_sum(gr, sumgr, dim=1)
            inv[0] = T.rsqrt(sumsq[0] / T.float32(hidden) + T.float32(eps))
            for j in T.Parallel(hidden):
                dx = gw_f[0, j] * inv[0] - r_f[0, j] * sumgr[0] * inv[0] * inv[0] * inv[0] / T.float32(hidden)
                out_s[0, j] = T.Cast(dtype, dx + T.Cast(T.float32, dr_s[0, j]))
                dw_s[0, j] = T.Cast(T.float32, dn_s[0, j]) * r_f[0, j] * inv[0]
            T.copy(out_s, DR[row, 0])
            T.copy(dw_s, DWPartial[row, 0])

    return main

