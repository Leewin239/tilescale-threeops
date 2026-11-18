"""Paired pure-kernel latency at the validated Qwen3-1.7B shapes."""

import argparse
import json
from statistics import median
import torch
import torch.distributed as dist
import tilelang
from tilelang.distributed.host import init_dist

from .config import PathConfig
from .runtime import LayerBuffers
from .kernels.ag import ag_gemm_sm_specialized_kernel
from .kernels.rs import gemm_rs_specialized_kernel
from .kernels.ar import gemm_ar_sm_specialized_kernel


def worker(rank, args):
    _, ranks, group = init_dist(rank, 2)
    torch.manual_seed(839 + rank)
    h = LayerBuffers(rank, ranks, group, PathConfig(), layer_count=1)
    if args.op == "ag":
        x = tilelang.tensor((h.local_M, h.H), torch.bfloat16, allocator=h.allocator).normal_(
            std=0.1
        )
        w = torch.randn((h.H, 2 * h.local_F), device="cuda", dtype=x.dtype) * 0.1
        output = h.raw
        candidate = ag_gemm_sm_specialized_kernel(
            h.M,
            2 * h.Fdim,
            h.H,
            ranks,
            128,
            256,
            64,
            256,
            12,
            4,
            True,
            grid_ctas=80,
            defer_tma_writeback=True,
        )
        baseline = h.ag
        inputs = (
            x,
            w,
            h.mcast_A,
            h.gathered_A,
            h.ag_mcast_signal,
            h.ag_local_signal,
            h.ag_barriers,
            output,
            4,
        )
        full = torch.empty((h.M, h.H), device="cuda", dtype=x.dtype)
        dist.all_gather_into_tensor(full, x, group=group)
        expected = full @ w
    elif args.op == "rs":
        x = torch.randn((h.M, h.local_F), device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn((h.local_F, h.H), device="cuda", dtype=x.dtype) * 0.1
        output = h.down
        baseline = h.rs
        candidate = gemm_rs_specialized_kernel(
            h.M,
            h.H,
            h.Fdim,
            ranks,
            128,
            128,
            128,
            256,
            12,
            3,
            True,
            grid_ctas=80,
            local_first=True,
        )
        inputs = x, w, output
        expected = torch.empty_like(output)
        dist.reduce_scatter_tensor(expected, x @ w, group=group)
    else:
        x = torch.randn((h.M, 1024), device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn((1024, h.H), device="cuda", dtype=x.dtype) * 0.1
        output = h.local_C
        baseline = h.ar
        candidate = gemm_ar_sm_specialized_kernel(
            h.M,
            h.H,
            1024,
            ranks,
            128,
            256,
            64,
            256,
            12,
            4,
            8192,
            grid_ctas=132,
            row_major=True,
            contiguous_reduction=True,
        )
        inputs = x, w, output, h.mcast_C, h.ar_mcast_signal, h.ar_local_signal, 16
        expected = x @ w
        dist.all_reduce(expected, group=group)
    candidate.initialize(allocator=h.allocator)

    def call(kernel):
        if args.op == "rs":
            output.zero_()
        elif args.op == "ar":
            h.ar_reset(h.ar_local_signal)
        h.barrier(h.pre_signal)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        kernel(*inputs)
        end.record()
        if args.op == "ag":
            h.ag_reset(h.ag_local_signal, h.ag_barriers, True)
        h.barrier(h.post_signal)
        torch.cuda.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end) * 1000, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        return elapsed.item()

    call(baseline)
    reference = output.clone()
    call(candidate)
    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.02)
    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.02)
    for _ in range(args.warmup):
        call(baseline)
        call(candidate)
    pairs = []
    for repeat in range(args.repeats):
        order = (
            (baseline, candidate, candidate, baseline)
            if repeat % 2 == 0
            else (candidate, baseline, baseline, candidate)
        )
        values = [call(kernel) for kernel in order]
        a, b = ((values[0] + values[3]) / 2, (values[1] + values[2]) / 2)
        if repeat % 2:
            a, b = b, a
        pairs.append((a, b))
    if rank == 0:
        print(
            json.dumps(
                {
                    "operator": args.op,
                    "timing": "kernel_only_us",
                    "validated": True,
                    "baseline_median": median(a for a, _ in pairs),
                    "optimized_median": median(b for _, b in pairs),
                    "paired_speedup": median(a / b for a, b in pairs),
                    "faster_pairs": sum(a > b for a, b in pairs),
                    "pairs": pairs,
                },
                sort_keys=True,
            )
        )
    h.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("ag", "rs", "ar"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("Expected nonnegative warmup and positive repeats")
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
