"""Merge the continued student and shard full-parameter training with FSDP."""

from contextlib import nullcontext
import torch
from torch import nn

from pixelone.model import LoRALinear
from pixelone.inputs import prepare
from pixelone.losses import support


def enable_full_training(model, recipe):
    experts_backend = recipe.get('experts_backend', 'eager')
    if experts_backend not in ('eager', 'grouped_mm'):
        raise ValueError('Unknown experts backend')
    if model.native.is_moe and experts_backend == 'grouped_mm':
        if not hasattr(torch.nn.functional, 'grouped_mm'):
            raise RuntimeError(
                'Grouped experts require the image PyTorch grouped_mm implementation'
            )
        for module in model.native.backbone.language_model.modules():
            if type(module).__name__ == 'Qwen3_5MoeExperts':
                module.config._experts_implementation = 'grouped_mm'
    merged = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        with torch.no_grad():
            weight = module.base.weight.float() + module.scale * (
                module.lora_B.float() @ module.lora_A.float()
            )
            module.base.weight = nn.Parameter(weight)
        parent, leaf = name.rsplit('.', 1)
        model.get_submodule(parent).__setattr__(leaf, module.base)
        merged.append(name)
    if not merged:
        raise ValueError('Full training requires a loaded 22k LoRA student before merging')
    model.native.head.bias = nn.Parameter(model.native.head.bias.detach().reshape(1).clone())
    model.requires_grad_(True)
    if not recipe['train_vision']:
        model.native.backbone.visual.requires_grad_(False)
    elif recipe.get('gradient_checkpointing', True):
        model.native.backbone.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
    model.native.train_vision = recipe['train_vision']
    model.native.full_parameter_training = True
    model.unified_initialization = dict(
        model.unified_initialization,
        training_mode='full',
        lora_merge='FP32 B @ A * alpha/rank merged into base projections',
        merged_lora_modules=merged,
        train_vision=recipe['train_vision'],
        trainable_parameters_before_sharding=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    )


class BatchPayload:
    def __init__(self, records):
        self.records = records


class FullBatchStudent(nn.Module):
    def __init__(self, student, processor, device):
        super().__init__()
        self.student, self.processor, self.device = student, processor, device
        self.timer = None

    def forward(self, payload):
        prepared = payload.records
        phase = self.timer.phase if self.timer is not None else lambda name: nullcontext()
        with phase('vision_and_processor'):
            for value in prepared:
                valid, positive = support(value['layout'], value['row']['bbox_xyxy'], self.device)
                group, features, actual, encoding = prepare(
                    self.student, self.processor, value.pop('image'), value['row']['instruction']
                )
                if actual != value['layout']:
                    raise ValueError('Full training changed the native image layout')
                value.update(
                    valid=valid,
                    positive=positive,
                    group=group,
                    features=features,
                    encoding=encoding,
                )
        with phase('language_forward'):
            outputs = self.student(
                [value['group'] for value in prepared],
                [value['features'] for value in prepared],
                [value['layout'] for value in prepared],
                [value['required'] for value in prepared],
                [value['positive'].any(-1) for value in prepared],
                self.student.recipe['routing_topk'],
            )
        return outputs, prepared


def wrap_full_training(model, processor, device, recipe, topology):
    batch_model = FullBatchStudent(model, processor, device)
    if device.type != 'cuda':
        model.float()
        return batch_model
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.device_mesh import init_device_mesh

    mode = recipe.get('sharding', 'full_shard')
    arguments = dict(device_id=device, use_orig_params=True, limit_all_gathers=True)
    if mode == 'hybrid_shard' and topology['nodes'] > 1:
        arguments['sharding_strategy'] = ShardingStrategy.HYBRID_SHARD
        arguments['device_mesh'] = init_device_mesh(
            'cuda',
            (topology['nodes'], topology['processes']),
            mesh_dim_names=('replicate', 'shard'),
        )
    elif mode in ('full_shard', 'hybrid_shard'):
        arguments['sharding_strategy'] = ShardingStrategy.FULL_SHARD
    else:
        raise ValueError('Unknown full-parameter sharding strategy')
    precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    language = model.native.backbone.language_model
    for index, layer in enumerate(language.layers):
        language.layers[index] = FSDP(layer.float(), mixed_precision=precision, **arguments)
    visual = model.native.backbone.visual
    if recipe['train_vision']:
        for index, block in enumerate(visual.blocks):
            visual.blocks[index] = FSDP(block.float(), mixed_precision=precision, **arguments)
    for module in model.modules():
        if not isinstance(module, FSDP):
            for parameter in module.parameters(recurse=False):
                if parameter.dtype == torch.bfloat16:
                    parameter.data = parameter.data.float()
    return FSDP(batch_model, **arguments)


def parameter_groups(model, recipe):
    groups = []
    for name, prefix, learning_rate in [
        ('language', 'native.backbone.language_model.', recipe['full_lr']),
        ('vision', 'native.backbone.visual.', recipe['vision_lr']),
        ('head', 'native.head.', recipe['head_lr']),
        ('spatial', 'spatial.', recipe['spatial_lr']),
    ]:
        parameters = [
            parameter
            for key, parameter in model.named_parameters()
            if key.startswith(prefix) and parameter.requires_grad
        ]
        if parameters:
            groups.append(
                dict(name=name, params=parameters, lr=learning_rate, base_lr=learning_rate)
            )
    selected = {id(parameter) for group in groups for parameter in group['params']}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if selected != expected:
        raise ValueError('Full optimizer groups do not cover every trainable parameter')
    return groups


def clip_full_gradients(wrapped, parameters, clip):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(wrapped, FSDP):
        norm = wrapped.clip_grad_norm_(clip)
    else:
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    if not bool(torch.isfinite(norm)):
        raise ValueError('Nonfinite full-model gradient norm')
    return float(norm), {}
