"""Pai-Megatron-Patch Qwen3 entry point for the eight path configurations."""

import os
import sys
import torch
import torch.distributed as dist

# Optional compatibility with a separate TileScale Python installation.
if os.environ.get("TILESCALE_PYTHON_PATH"):
    sys.path.extend(os.environ["TILESCALE_PYTHON_PATH"].split(os.pathsep))

from megatron.core.enums import ModelType
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training import get_args, get_tokenizer, pretrain
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import train_valid_test_datasets_provider
from megatron_patch.template.helper import forward_step
from megatron_patch import tokenizer as pai_tokenizer
from pretrain_qwen import model_provider as native_model_provider

from .config import PathConfig
from .megatron import install_layer
from .timing import time_layer, report_layer_times, time_model_phases, report_model_phases

shared_buffers = []


def model_provider(*args, **kwargs):
    model = native_model_provider(*args, **kwargs)
    layers = [layer for layer in model.modules() if isinstance(layer, TransformerLayer)]
    expected = list(range(1, get_args().num_layers + 1))
    if sorted(layer.layer_number for layer in layers) != expected or len(layers) != 28:
        raise ValueError("Expected all 28 Qwen3-1.7B decoder layers on one pipeline stage")
    config = PathConfig.from_name(os.environ.get("THREEOPS_CONFIG", "paths_ag_rs_ar"))
    checks = os.environ.get("THREEOPS_CHECK", "1") == "1"
    if dist.get_rank() == 0:
        print(f"TILELANG_ABLATION_LAYERS count={len(layers)}", flush=True)
    for layer in layers:
        install_layer(layer, config, shared_buffers, checks=checks, layer_count=len(layers))
        if layer.layer_number == 14:
            time_layer(layer)
    return model


def dataset_provider(sample_counts):
    pai_tokenizer._GLOBAL_TOKENIZER = get_tokenizer()
    return train_valid_test_datasets_provider(sample_counts)


def main():
    dataset_provider.is_distributed = True
    time_model_phases()
    try:
        pretrain(
            dataset_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            extra_args_provider=get_patch_args,
        )
        report_layer_times()
        report_model_phases()
    finally:
        if shared_buffers:
            torch.cuda.synchronize()
            shared_buffers[0].close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
