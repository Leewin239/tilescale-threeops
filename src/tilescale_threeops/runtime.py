"""Distributed buffers and forward calls for the retained experiment."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.distributed import allocator as allocator_module
from tilelang.distributed.allocator import get_allocator

from .kernels.ag import ag_gemm_sm_specialized_kernel, reset_ag_gemm_specialized_state_kernel
from .kernels.rs import gemm_rs_specialized_kernel
from .kernels.ar import gemm_ar_sm_specialized_kernel as optimized_ar
from .kernels.baseline_ar import gemm_ar_sm_specialized_kernel as baseline_ar
from .kernels.baseline_ar import reset_gemm_ar_specialized_state_kernel
from .vmm import vmm_tensor_view


@tilelang.jit(compile_once=True)
def rank_barrier_kernel(ranks):
    @T.prim_func
    def main(barrier: T.Tensor((ranks,), T.int32)):
        with T.Kernel(1, threads=32):
            T.barrier_blocks(barrier)
            T.fence_sys()

    return main


class LayerBuffers:
    """Per-layer saved input and shared forward scratch for one microbatch."""

    def __init__(self, rank, ranks, group, config, owner=None, layer_count=28):
        self.rank, self.ranks, self.group = rank, ranks, group
        self.config = config
        self.M, self.H, self.Fdim = 4096, 2048, 6144
        self.local_M, self.local_F = self.M // ranks, self.Fdim // ranks
        tokens, hidden, ffn = self.M, self.H, self.Fdim
        self.owns_allocator = owner is None
        # Preserve the long-run allocation budget, with extra scratch headroom.
        original = allocator_module.tensor_from_ptr

        def wrap_ptr(ptr, shape, dtype, device, take_ownership=False, _owner=None):
            try:
                return original(ptr, shape, dtype, device, take_ownership, _owner=_owner)
            except RuntimeError as error:
                if take_ownership or "CUDA error: invalid argument" not in str(error):
                    raise
                return vmm_tensor_view(ptr, shape, dtype, device, _owner)

        allocator_module.tensor_from_ptr = wrap_ptr
        try:
            self.allocator = (
                owner.allocator
                if owner
                else get_allocator(
                    size=2**28 + 2 * tokens * (layer_count * (ffn // ranks + 2 * hidden) + ffn),
                    device=f"cuda:{rank}",
                    is_distributed=True,
                    local_rank=rank,
                    num_local_ranks=ranks,
                    group=group,
                    use_vmm=True,
                    mcast_size=2 * tokens * hidden * (layer_count + 1) + 65536 * layer_count,
                )
            )
            self._allocate(owner)
        finally:
            allocator_module.tensor_from_ptr = original

        self.ag = ag_gemm_sm_specialized_kernel(
            tokens,
            2 * ffn,
            hidden,
            ranks,
            128,
            256,
            64,
            256,
            12,
            4,
            True,
            grid_ctas=80,
            defer_tma_writeback=config.ag,
        )
        self.ag_reset = reset_ag_gemm_specialized_state_kernel(tokens // 256, ranks, 256)
        self.rs = gemm_rs_specialized_kernel(
            tokens,
            hidden,
            ffn,
            ranks,
            128,
            128 if config.rs else 256,
            128 if config.rs else 64,
            256,
            12,
            3 if config.rs else 4,
            True,
            grid_ctas=80,
            local_first=config.rs,
        )
        self.ar = (
            optimized_ar(
                tokens,
                hidden,
                1024,
                ranks,
                128,
                256,
                64,
                256,
                12,
                4,
                8192,
                grid_ctas=torch.cuda.get_device_properties(rank).multi_processor_count,
                row_major=True,
                contiguous_reduction=True,
            )
            if config.ar
            else baseline_ar(
                tokens,
                hidden,
                1024,
                ranks,
                128,
                256,
                64,
                256,
                12,
                4,
                4096,
            )
        )
        self.ar_reset = reset_gemm_ar_specialized_state_kernel(tokens // 128 * (hidden // 256), 256)
        self.barrier = rank_barrier_kernel(ranks)
        for kernel in (self.ag, self.ag_reset, self.rs, self.ar, self.ar_reset, self.barrier):
            kernel.initialize(allocator=self.allocator)
        self.ag_reset(self.ag_local_signal, self.ag_barriers, False)
        torch.cuda.synchronize()
        dist.barrier(group)

    def _allocate(self, owner):
        def tensor(shape, dtype=torch.bfloat16):
            return tilelang.tensor(shape, dtype, allocator=self.allocator)

        multicast, local = self.allocator._allocate_mcast_tensor((self.M * self.H,), torch.bfloat16)
        self.mcast_A, self.gathered_A = multicast.view(self.M, self.H), local.view(self.M, self.H)
        self.ag_mcast_signal, self.ag_local_signal = self.allocator._allocate_mcast_tensor(
            (self.M // 256,),
            torch.uint32,
        )
        if owner:
            self.mcast_C, self.local_C = owner.mcast_C, owner.local_C
            self.ar_mcast_signal, self.ar_local_signal = (
                owner.ar_mcast_signal,
                owner.ar_local_signal,
            )
            self.raw, self.down = owner.raw, owner.down
        else:
            multicast, local = self.allocator._allocate_mcast_tensor(
                (self.M * self.H,), torch.bfloat16
            )
            self.mcast_C, self.local_C = multicast.view(self.M, self.H), local.view(self.M, self.H)
            self.ar_mcast_signal, self.ar_local_signal = self.allocator._allocate_mcast_tensor(
                (self.M // 128 * (self.H // 256),),
                torch.uint32,
            )
            self.raw = tensor((self.M, 2 * self.local_F))
            self.down = tensor((self.local_M, self.H))
        self.ag_barriers = tensor((2, self.ranks), torch.int32).zero_()
        self.pre_signal = tensor((self.ranks,), torch.int32).zero_()
        self.post_signal = tensor((self.ranks,), torch.int32).zero_()

    def mlp_projection(self, x, weight_kn):
        if self.config.rs:
            # The AG's rank rendezvous orders this clear before RS writes.
            self.down.zero_()
        raw = torch.empty_like(self.raw) if self.config.rs else self.raw
        self.barrier(self.pre_signal)
        self.ag(
            x,
            weight_kn,
            self.mcast_A,
            self.gathered_A,
            self.ag_mcast_signal,
            self.ag_local_signal,
            self.ag_barriers,
            raw,
            4,
        )
        self.ag_reset(self.ag_local_signal, self.ag_barriers, True)
        if self.config.ag:
            begin = self.rank * self.local_M
            self.gathered_A[begin : begin + self.local_M].copy_(x)
        gate, up = raw.split(self.local_F, dim=1)
        return F.silu(gate) * up, raw

    def mlp_scatter(self, activation, weight_kn, residual):
        if not self.config.rs:
            self.down.zero_()
            self.barrier(self.pre_signal)
        self.rs(activation, weight_kn, self.down)
        self.barrier(self.post_signal)
        return self.down + residual

    def attention_projection(self, x, weight_kn):
        self.ar_reset(self.ar_local_signal)
        self.barrier(self.pre_signal)
        self.ar(
            x, weight_kn, self.local_C, self.mcast_C, self.ar_mcast_signal, self.ar_local_signal, 16
        )
        self.barrier(self.post_signal)
        begin = self.rank * self.local_M
        local = self.local_C[begin : begin + self.local_M]
        return local if self.config.ar else local.clone()

    def close(self):
        if self.owns_allocator:
            self.allocator.close()
