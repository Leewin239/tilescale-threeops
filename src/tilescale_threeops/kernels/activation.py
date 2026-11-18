# Adapted from TileScale. See third_party/LICENSE.TileScale.
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver

@tilelang.jit(compile_once=True)
def swiglu_backward_packed(rows, cols, dtype=T.bfloat16):
    assert rows % 4 == 0 and cols % 128 == 0

    @T.prim_func
    def main(
        GateUp: T.Tensor((rows, 2 * cols), dtype),
        DAct: T.Tensor((rows, cols), dtype),
        DGateUp: T.Tensor((rows, 2 * cols), dtype),
    ):
        with T.Kernel(rows // 4, cols // 128, threads=128) as (by, bx):
            for i, j in T.Parallel(4, 128):
                row = by * 4 + i
                col = bx * 128 + j
                gate = T.cast(GateUp[row, col], T.float32)
                up = T.cast(GateUp[row, cols + col], T.float32)
                dact = T.cast(DAct[row, col], T.float32)
                sigmoid = 1.0 / (1.0 + T.exp(-gate))
                DGateUp[row, col] = T.cast(dact * up * sigmoid *
                                          (1.0 + gate * (1.0 - sigmoid)), dtype)
                DGateUp[row, cols + col] = T.cast(dact * gate * sigmoid, dtype)

    return main

