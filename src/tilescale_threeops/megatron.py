"""Install the retained paths on Pai-Megatron Qwen3 TransformerLayer objects."""

from functools import wraps
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from transformer_engine.pytorch.module.rmsnorm import _RMSNorm

from .autograd import MLPFunction, ProjectionFunction, ResidualNormFunction
from .runtime import LayerBuffers


def te_rmsnorm(x, fc1):
    return _RMSNorm.apply(
        x.contiguous(),
        fc1.layer_norm_weight,
        fc1.eps,
        fc1.fwd_ln_sm_margin,
        fc1.bwd_ln_sm_margin,
        fc1.inf_ln_sm_margin,
        fc1.zero_centered_gamma,
        torch.is_grad_enabled(),
        x.dtype,
    )


def install_layer(layer, config, shared_buffers, *, checks=True, layer_count=28):
    args = get_args()
    group = parallel_state.get_tensor_model_parallel_group()
    rank, ranks = dist.get_rank(group), dist.get_world_size(group)
    if (
        ranks != 2
        or dist.get_world_size() != 2
        or rank != torch.cuda.current_device()
        or args.transformer_impl != "transformer_engine"
        or not args.sequence_parallel
        or not args.swiglu
        or not args.bf16
        or args.normalization != "RMSNorm"
        or args.micro_batch_size != 1
        or args.global_batch_size != 1
        or args.hidden_dropout
        or args.attention_dropout
        or args.recompute_granularity is not None
        or args.gradient_accumulation_fusion
        or args.add_bias_linear
        or (args.seq_length, args.hidden_size, args.ffn_hidden_size) != (4096, 2048, 6144)
    ):
        raise ValueError("Expected the validated Qwen3-1.7B TP2/SP BF16 single-microbatch workload")
    device = torch.cuda.get_device_properties(rank)
    if device.major < 9 or device.multi_processor_count != 132:
        raise ValueError("This release validates H100 with 132 SMs")
    fc1, fc2 = layer.mlp.linear_fc1, layer.mlp.linear_fc2
    proj = layer.self_attention.linear_proj
    if (
        fc1.weight.shape != (6144, 2048)
        or fc2.weight.shape != (2048, 3072)
        or proj.weight.shape != (2048, 1024)
        or fc1.eps != 1e-6
        or fc1.zero_centered_gamma
        or isinstance(layer.self_attn_bda, torch.nn.Module)
    ):
        raise ValueError("Expected bias-free Qwen3 weights and functional residual addition")
    native_mlp, native_projection, native_bda = (
        layer._forward_mlp,
        proj.forward,
        layer.self_attn_bda,
    )
    h = None
    pending_mlp, pending_proj, pending_norm = checks, checks, checks
    cached_norm = None

    def get_buffers():
        nonlocal h
        if h is None:
            owner = shared_buffers[0] if shared_buffers else None
            h = LayerBuffers(rank, ranks, group, config, owner, layer_count)
            if owner is None:
                shared_buffers.append(h)
            if rank == 0:
                print(f"TILELANG_THREEOP_ABLATION mode={config.name} joint_rs=0", flush=True)
                print(
                    f"TILELANG_MLP layer={layer.layer_number} tokens=4096 tp=2 "
                    f"ag_grid=80 rs_grid=80 ag_comm=4 qkv_grid=0 proj_grid=80 "
                    f"qkv_comm=4 packed_ag={int(config.ag)}",
                    flush=True,
                )
        return h

    def check(label, native, replacement, x, weights):
        probe = x.detach().clone().requires_grad_(True)
        expected = native(probe)
        upstream = torch.randn(
            expected.shape,
            device=x.device,
            dtype=x.dtype,
            generator=torch.Generator(device=x.device).manual_seed(101 + rank),
        )
        expected_grad = torch.autograd.grad(expected, (probe, *weights), upstream)
        actual = replacement(probe)
        actual_grad = torch.autograd.grad(actual, (probe, *weights), upstream)
        errors = torch.stack(
            [
                (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                for a, b in zip((actual, *actual_grad), (expected, *expected_grad))
            ]
        ).nan_to_num(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        dist.all_reduce(errors, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(
                f"TILELANG_{label}_CHECK layer={layer.layer_number} relative_errors={errors.tolist()}",
                flush=True,
            )
        if errors[0] > 0.03 or (errors[1:] > 0.1).any():
            raise RuntimeError(f"{label} output/gradient reference check failed")

    def eligible(x, rows):
        return (
            layer.training
            and x.shape == (rows, 1, 2048 if rows == 2048 else 1024)
            and x.dtype == torch.bfloat16
        )

    def run_mlp(x):
        nonlocal cached_norm
        residual = x.reshape(-1, 2048).contiguous()
        if cached_norm is not None and cached_norm[0] is x:
            normed = cached_norm[1]
            cached_norm = None
        else:
            normed = te_rmsnorm(x, fc1)
        output = MLPFunction.apply(
            normed.reshape_as(residual).contiguous(),
            residual,
            fc1.weight,
            fc2.weight,
            get_buffers(),
        ).view_as(x)
        return make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

    @wraps(native_mlp)
    def forward_mlp(x, inference_context=None):
        nonlocal pending_mlp
        if inference_context is not None or not eligible(x, 2048):
            return native_mlp(x, inference_context)
        get_buffers()
        if pending_mlp:
            check("MLP", native_mlp, run_mlp, x, (fc1.weight, fc2.weight, fc1.layer_norm_weight))
            pending_mlp = False
        return run_mlp(x)

    def run_projection(x):
        return ProjectionFunction.apply(
            x.reshape(-1, 1024).contiguous(), proj.weight, get_buffers()
        ).view(-1, 1, 2048)

    @wraps(native_projection)
    def forward_projection(x, *args, **kwargs):
        nonlocal pending_proj
        if args or kwargs or not eligible(x, 4096):
            return native_projection(x, *args, **kwargs)
        get_buffers()
        if pending_proj:
            check("PROJ", lambda x: native_projection(x)[0], run_projection, x, (proj.weight,))
            pending_proj = False
        return run_projection(x), None

    def residual_factory(training, use_fusion):
        native_add = native_bda(training, use_fusion)

        def add_and_norm(attention, residual, dropout):
            nonlocal cached_norm, pending_norm
            projected, bias = attention
            if not training or dropout or bias is not None or not eligible(residual, 2048):
                return native_add(attention, residual, dropout)
            if cached_norm is not None:
                raise RuntimeError("Unconsumed attention normalization")
            inputs = projected.reshape(-1, 2048), residual.reshape(-1, 2048), fc1.layer_norm_weight
            combined, normed = ResidualNormFunction.apply(*inputs)
            combined, normed = combined.view_as(residual), normed.view_as(residual)
            if pending_norm:
                p = projected.detach().requires_grad_(True)
                r = residual.detach().requires_grad_(True)
                expected = native_add((p, None), r, dropout)
                expected_norm = te_rmsnorm(expected, fc1)
                torch.testing.assert_close(combined, expected, atol=0.0078125, rtol=0.02)
                torch.testing.assert_close(normed, expected_norm, atol=0.015625, rtol=0.04)
                actual = ResidualNormFunction.apply(
                    p.reshape(-1, 2048), r.reshape(-1, 2048), fc1.layer_norm_weight
                )
                upstream = tuple(torch.cos(value.detach()) for value in (expected, expected_norm))
                reference_grad = torch.autograd.grad(
                    (expected, expected_norm), (p, r, fc1.layer_norm_weight), upstream
                )
                actual_grad = torch.autograd.grad(
                    actual,
                    (p, r, fc1.layer_norm_weight),
                    tuple(g.reshape(-1, 2048) for g in upstream),
                )
                errors = [
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                    for a, b in zip(actual_grad, reference_grad)
                ]
                if not all(torch.isfinite(e) and e <= 0.02 for e in errors):
                    raise RuntimeError(f"Residual/norm gradient mismatch {errors}")
                if rank == 0:
                    print(
                        f"TILELANG_RESNORM_CHECK layer={layer.layer_number} relative_errors={[e.item() for e in errors]}",
                        flush=True,
                    )
                pending_norm = False
            cached_norm = combined, normed
            return combined

        return add_and_norm

    layer._forward_mlp = forward_mlp
    proj.forward = forward_projection
    if config.ar:
        layer.self_attn_bda = residual_factory
