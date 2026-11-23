"""Temporary attention/kernel policies and inference precision settings."""

from contextlib import ExitStack, contextmanager
from importlib import import_module

import torch

OPTIONAL_KERNELS = (
    'FusedRMSNormGated',
    'causal_conv1d_fn',
    'causal_conv1d_update',
    'chunk_gated_delta_rule',
    'fused_recurrent_gated_delta_rule',
)


@contextmanager
def qwen_kernel_backend(backend='torch', module=None):
    """Select native or torch kernels and restore temporary overrides on exit."""
    if backend not in ('torch', 'native'):
        raise ValueError('Choose torch or native Qwen kernels')
    module = module or import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    original = {name: getattr(module, name) for name in OPTIONAL_KERNELS}
    runtime = dict(
        backend=backend,
        optional_kernels_available={name: value is not None for name, value in original.items()},
        optional_kernels_enabled={
            name: backend == 'native' and value is not None for name, value in original.items()
        },
    )
    if backend == 'native':
        yield runtime
        return
    original['is_fast_path_available'] = module.is_fast_path_available
    try:
        for name in OPTIONAL_KERNELS:
            setattr(module, name, None)
        module.is_fast_path_available = False
        yield runtime
    finally:
        for name, value in original.items():
            setattr(module, name, value)


def bidirectional_mask(config, inputs_embeds, attention_mask=None, past_key_values=None, **kwargs):
    """Mask padding only in existing full-attention layers, without a causal triangle."""
    if past_key_values is not None:
        raise ValueError('Bidirectional spatial training does not support autoregressive KV caches')
    if config._attn_implementation not in ('sdpa', 'eager'):
        raise ValueError('Bidirectional full layers currently require SDPA or eager attention')
    batch, length = inputs_embeds.shape[:2]
    if attention_mask is None:
        valid = torch.ones((batch, length), device=inputs_embeds.device, dtype=torch.bool)
    else:
        if attention_mask.shape != (batch, length):
            raise ValueError('Expected independent per-sample 2D padding mask')
        valid = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
    mask = torch.zeros(
        (batch, 1, 1, length), device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    return mask.masked_fill(~valid[:, None, None, :], torch.finfo(inputs_embeds.dtype).min)


@contextmanager
def full_layer_attention(module, mode):
    if mode == 'causal':
        yield
        return
    if mode != 'bidirectional_full_layers':
        raise ValueError('Unknown attention mode: ' + mode)
    original = module.create_causal_mask
    module.create_causal_mask = bidirectional_mask
    try:
        yield
    finally:
        module.create_causal_mask = original


@contextmanager
def native_runtime(backend='torch', required_model=None, attention_mode='causal'):
    """Apply the requested attention and kernel policy to dense and MoE backbones."""
    with ExitStack() as stack:
        for name in ('qwen3_5', 'qwen3_5_moe'):
            module = import_module(f'transformers.models.{name}.modeling_{name}')
            stack.enter_context(full_layer_attention(module, attention_mode))
            runtime = stack.enter_context(qwen_kernel_backend(backend, module))
            if backend == 'native' and (required_model is None or name == required_model):
                missing = [
                    key
                    for key, enabled in runtime['optional_kernels_enabled'].items()
                    if not enabled
                ]
                if missing:
                    raise RuntimeError(
                        'Native Qwen kernels unavailable: '
                        + ', '.join(missing)
                        + '. Install compatible flash-linear-attention and causal-conv1d in the active Python environment; '
                        'use kernel_backend=torch only for an explicitly labelled baseline.'
                    )
            cls = getattr(
                module, 'Qwen3_5DecoderLayer' if name == 'qwen3_5' else 'Qwen3_5MoeDecoderLayer'
            )
            if not hasattr(cls, 'block_type'):
                cls.block_type = property(lambda layer: layer.layer_type)
        yield


def cast_training_inputs(module, args, kwargs):
    from torch.utils._pytree import tree_map

    def cast(value):
        return (
            value.to(torch.bfloat16)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        )

    return tree_map(cast, args), tree_map(cast, kwargs)


def inference_precision(model, recipe, precision):
    """Restore the reference BF16 compute partition while retaining FP32 parameters."""
    if precision != 'training_cuda':
        return
    blocks = list(model.native.backbone.language_model.layers)
    if recipe['train_vision']:
        blocks += list(model.native.backbone.visual.blocks)
    for block in blocks:
        block.register_forward_pre_hook(cast_training_inputs, with_kwargs=True)
        for module in block.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    setattr(module, name, buffer.to(torch.bfloat16))
