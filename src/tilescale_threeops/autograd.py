"""Training rules retained from the validated forward/backward paths."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .kernels.activation import swiglu_backward_packed
from .kernels.residual_norm import ar_residual_rmsnorm_kernel, ar_residual_rmsnorm_backward_kernel


class MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, fc1_weight, fc2_weight, buffers):
        activation, gate_up = buffers.mlp_projection(x, fc1_weight.T.contiguous())
        output = buffers.mlp_scatter(activation, fc2_weight.T.contiguous(), residual)
        ctx.buffers = buffers
        ctx.save_for_backward(
            x,
            fc1_weight,
            fc2_weight,
            gate_up if buffers.config.rs else None,
            activation if buffers.config.ag or buffers.config.rs else None,
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        x, fc1_weight, fc2_weight, gate_up, activation = ctx.saved_tensors
        h = ctx.buffers
        grad = grad.contiguous()
        full_x = (
            h.gathered_A
            if h.config.ag
            else torch.empty(
                (h.M, h.H),
                device=x.device,
                dtype=x.dtype,
            )
        )
        if not h.config.ag:
            dist.all_gather_into_tensor(full_x, x.contiguous(), group=h.group)
        full_grad = torch.empty((h.M, h.H), device=grad.device, dtype=grad.dtype)
        gather = dist.all_gather_into_tensor(full_grad, grad, group=h.group, async_op=h.config.rs)

        if h.config.ag or h.config.rs:
            gate_up = gate_up if gate_up is not None else full_x @ fc1_weight.T
            if h.config.rs:
                local_rows = slice(h.rank * h.local_M, (h.rank + 1) * h.local_M)
                peer_rank = 1 - h.rank
                peer_rows = slice(peer_rank * h.local_M, (peer_rank + 1) * h.local_M)
                local_dw2 = grad.T @ activation[local_rows]
                gather.wait()
            dact = full_grad @ fc2_weight
            dgate_up = torch.empty_like(gate_up)
            swiglu_backward_packed(h.M, h.local_F)(gate_up, dact, dgate_up)
            dx = torch.empty_like(x)
            if h.config.rs:
                dx_full = dgate_up @ fc1_weight
                scatter = dist.reduce_scatter_tensor(dx, dx_full, group=h.group, async_op=True)
            dw1 = dgate_up.T @ full_x
            dw2 = (
                torch.addmm(local_dw2, full_grad[peer_rows].T, activation[peer_rows])
                if h.config.rs
                else full_grad.T @ activation
            )
            if h.config.rs:
                scatter.wait()
            else:
                dist.reduce_scatter_tensor(dx, dgate_up @ fc1_weight, group=h.group)
        else:
            # Original TileScale MLP recomputes its local FFN for backward.
            gate, up = fc1_weight.chunk(2, dim=0)
            with torch.enable_grad():
                probe = full_x.detach().requires_grad_(True)
                wg = gate.T.contiguous().detach().requires_grad_(True)
                wu = up.T.contiguous().detach().requires_grad_(True)
                wd = fc2_weight.T.contiguous().detach().requires_grad_(True)
                partial = (F.silu(probe @ wg) * (probe @ wu)) @ wd
                dx_full, dwg, dwu, dwd = torch.autograd.grad(
                    partial, (probe, wg, wu, wd), full_grad
                )
            dw1 = torch.cat((dwg.T, dwu.T), dim=0)
            dw2 = dwd.T
            dx = torch.empty_like(x)
            dist.reduce_scatter_tensor(dx, dx_full.contiguous(), group=h.group)
        return dx, grad, dw1, dw2, None


class ProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, buffers):
        ctx.buffers = buffers
        weight_kn = weight.T.contiguous()
        ctx.save_for_backward(x, weight_kn)
        return buffers.attention_projection(x, weight_kn)

    @staticmethod
    def backward(ctx, grad):
        x, weight_kn = ctx.saved_tensors
        h = ctx.buffers
        full_grad = torch.empty((h.M, h.H), device=grad.device, dtype=grad.dtype)
        dist.all_gather_into_tensor(full_grad, grad.contiguous(), group=h.group)
        dw = x.T @ full_grad
        dx = full_grad @ weight_kn.T
        return dx, dw.T, None


class ResidualNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, residual, weight):
        rows, hidden = residual.shape
        combined, normed = torch.empty_like(residual), torch.empty_like(residual)
        ar_residual_rmsnorm_kernel(rows, hidden)(
            projected.contiguous(),
            residual,
            weight,
            combined,
            normed,
        )
        ctx.save_for_backward(combined, weight)
        return combined, normed

    @staticmethod
    def backward(ctx, grad_combined, grad_normed):
        residual, weight = ctx.saved_tensors
        rows, hidden = residual.shape
        dx = torch.empty_like(residual)
        partial = torch.empty((rows, hidden), dtype=torch.float32, device=residual.device)
        ar_residual_rmsnorm_backward_kernel(rows, hidden)(
            residual,
            weight,
            grad_normed.contiguous(),
            grad_combined.contiguous(),
            dx,
            partial,
        )
        return dx, dx, partial.sum(dim=0).to(weight.dtype)
