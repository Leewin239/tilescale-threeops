"""CUDA event timing used by the validated Megatron runs."""
from functools import wraps
import torch
import torch.distributed as dist

layer_samples = []
model_phase_samples = {"forward": [], "backward": []}

def time_layer(layer):
    original = layer.forward

    @wraps(original)
    def timed(*args, **kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        if not layer.training or not torch.is_grad_enabled() or not hidden.requires_grad:
            return original(*args, **kwargs)
        start, end, grad_start, grad_end = (
            torch.cuda.Event(enable_timing=True) for _ in range(4)
        )
        start.record()
        output, context = original(*args, **kwargs)
        end.record()

        def record_gradient(event):
            def hook(grad):
                event.record()
                return grad
            return hook

        output.register_hook(record_gradient(grad_start))
        hidden.register_hook(record_gradient(grad_end))
        layer_samples.append((start, end, grad_start, grad_end))
        return output, context

    layer.forward = timed


def report_layer_times():
    if not layer_samples:
        return
    torch.cuda.synchronize()
    times = torch.tensor([[a.elapsed_time(b), c.elapsed_time(d)]
                          for a, b, c, d in layer_samples], device="cuda")
    worst_total = times.sum(dim=1)
    dist.all_reduce(worst_total, op=dist.ReduceOp.MAX)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        for step, ((forward_ms, backward_ms), layer_ms) in enumerate(
                zip(times.tolist(), worst_total.tolist())):
            print(f"MEGATRON_LAYER_TIME sample={step} forward_ms={forward_ms:.6f} "
                  f"backward_ms={backward_ms:.6f} layer_ms={layer_ms:.6f}", flush=True)


def time_model_phases():
    from megatron.core.pipeline_parallel import schedules

    def wrap_phase(method, samples):
        @wraps(method)
        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = method(*args, **kwargs)
            end.record()
            samples.append((start, end))
            return result
        return timed

    schedules.forward_step = wrap_phase(schedules.forward_step, model_phase_samples["forward"])
    schedules.backward_step = wrap_phase(schedules.backward_step, model_phase_samples["backward"])


def report_model_phases():
    forward = model_phase_samples["forward"]
    backward = model_phase_samples["backward"]
    if len(forward) != len(backward):
        raise RuntimeError("Whole-model timing requires paired training forwards and backwards")
    torch.cuda.synchronize()
    times = torch.tensor(
        [[a.elapsed_time(b), c.elapsed_time(d)] for (a, b), (c, d) in zip(forward, backward)],
        device="cuda",
    )
    totals = times.sum(dim=1)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    dist.all_reduce(totals, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        for step, ((forward_ms, backward_ms), total_ms) in enumerate(zip(times.tolist(), totals.tolist())):
            print(f"MEGATRON_MODEL_TIME sample={step} forward_ms={forward_ms:.6f} "
                  f"backward_ms={backward_ms:.6f} model_ms={total_ms:.6f}", flush=True)

