"""Train T3 teachers, LoRA students, and full-parameter Pro continuations."""

import copy
import json
import math
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.distributed as distributed

from pixelone.model import Teacher, tensor_hash
from pixelone.runtime import native_runtime
from pixelone.states import StateSampler, prepare_state, read_image
from pixelone.geometry import Layout
from pixelone.distillation import project, selected_parents
from pixelone.full_training import (
    BatchPayload,
    FullBatchStudent,
    clip_full_gradients,
    enable_full_training,
    parameter_groups,
    wrap_full_training,
)
from pixelone.model import (
    ClickStudent,
    fresh_head,
    initialize_continued_student,
    initialize_student,
)
from pixelone.losses import unified_loss
from pixelone.checkpoint import load_checkpoint, save_checkpoint
from pixelone.data import GroundingDataset, IndexedTeacherCache
from pixelone.io import atomic_json, file_sha256
from pixelone.sampling import SampleStream
from pixelone.teacher import SearchTeacher, visited_states


def load_native(path, device):
    from transformers import AutoProcessor
    from pixelone.processor import LegacyProcessorAdapter
    from videojev_hf.modeling_videojev import VideoJevForVideoUnderstanding

    native = VideoJevForVideoUnderstanding.from_pretrained(
        path,
        dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        device_map={'': str(device)},
        attn_implementation='sdpa',
    )
    processor = LegacyProcessorAdapter(AutoProcessor.from_pretrained(path, trust_remote_code=False))
    return native, processor


def initialize_training_student(native, recipe, initialization, adapter=None):
    """Dispatch base, teacher, or continued-student initialization."""
    from safetensors.torch import load_file

    if initialization == 'base':
        if adapter:
            raise ValueError('Base initialization must not load a trained adapter')
        student = ClickStudent(native, recipe)
        student.spatial.load_state_dict(fresh_head(native.head.q.in_features, recipe).state_dict())
        student.unified_initialization = {'kind': 'base'}
        return student
    if not adapter:
        raise ValueError('Teacher/student initialization requires an adapter file')
    state = load_file(adapter, device='cpu')
    if initialization == 'teacher':
        return initialize_student(native, recipe, state)[0]
    if initialization != 'student':
        raise ValueError('Initialization must be base, teacher or student')
    spatial = {
        key.removeprefix('spatial.'): value
        for key, value in state.items()
        if key.startswith('spatial.')
    }
    expected = {'adapter_tensor_sha256': tensor_hash(state), 'spatial_sha256': tensor_hash(spatial)}
    return initialize_continued_student(native, recipe, state, expected)[0]


def adapter_parameter_groups(model, recipe):
    groups = []
    for label, rate in [
        ('lora', recipe['lora_lr']),
        ('head', recipe['head_lr']),
        ('spatial', recipe.get('spatial_lr', recipe['head_lr'])),
    ]:
        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (
                ('.lora_' in name and label == 'lora')
                or (name.startswith('native.head.') and label == 'head')
                or (name.startswith('spatial.') and label == 'spatial')
            )
        ]
        if parameters:
            groups.append({'params': parameters, 'lr': rate, 'base_lr': rate, 'name': label})
    if {id(parameter) for group in groups for parameter in group['params']} != {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }:
        raise ValueError('Optimizer does not cover the trainable adapter parameters')
    return groups


def schedule_rate(step, steps, recipe):
    warmup = recipe.get('lr_warmup_steps', 0)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if recipe.get('lr_schedule', 'constant') == 'constant':
        return 1.0
    if recipe['lr_schedule'] != 'cosine':
        raise ValueError('Unsupported learning-rate schedule')
    progress = (step - warmup) / max(1, steps - warmup - 1)
    minimum = recipe.get('min_lr_ratio', 0.1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def student_batch(student, rows, stream, positions, teacher_cache):
    """Prepare sampled screenshots, GT support, and optional teacher supervision."""
    payload = []
    for position in positions:
        row = rows[stream.index(position)]
        layout = Layout(*row['image_size'])
        target = layout.center_target(row['bbox_xyxy'])
        soft = project(teacher_cache.get(row), row['bbox_xyxy'], layout) if teacher_cache else None
        payload.append(
            {
                'row': row,
                'layout': layout,
                'target': target,
                'soft': soft,
                'required': selected_parents(target, soft),
                'image': read_image(row),
            }
        )
    return BatchPayload(payload)


def run_training(configuration, *, resume=None):
    """Run the configured teacher/student stage, including resume and checkpoint saving."""
    config = copy.deepcopy(configuration)
    recipe = config['recipe']
    stage = config['stage']
    if stage not in ('teacher', 'student'):
        raise ValueError('Stage must be teacher or student')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    if world > 1:
        distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    seed = recipe['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rows = GroundingDataset(config['data'], config.get('image_root'))
    if config.get('expected_targets') is not None and len(rows) != config['expected_targets']:
        raise ValueError('Training target count differs from the configured dataset')
    full = stage == 'student' and recipe.get('train_mode') == 'full'
    if full and config['initialization'] != 'student':
        raise ValueError('Pro continuation must retain a trained student pixel head')
    batch, micro = recipe['global_batch'], recipe['micro_batch']
    if batch <= 0 or micro <= 0 or batch % (world * micro):
        raise ValueError('Global batch must be divisible by world size times micro batch')
    steps = (
        recipe['supervised_steps'] + recipe['mixed_steps']
        if stage == 'teacher'
        else config.get('steps') or math.ceil(len(rows) * config.get('epochs', 1) / batch)
    )
    root = Path(config['output'])
    if rank == 0:
        if root.exists() and any(root.iterdir()) and not resume:
            raise ValueError('Nonempty output directory requires an explicit resume checkpoint')
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / 'configuration.json', config)
    if world > 1:
        distributed.barrier()
    cache = None
    model = None
    try:
        with native_runtime(
            recipe['kernel_backend'], attention_mode=recipe.get('attention_mode', 'causal')
        ):
            native, processor = load_native(config['base_model'], device)
            assets = root / 'assets'
            if rank == 0:
                processor.processor.save_pretrained(assets)
                native.config.to_json_file(assets / 'config.json', use_diff=False)
            if world > 1:
                distributed.barrier()
            asset_hashes = {
                path.name: file_sha256(path) for path in sorted(assets.iterdir()) if path.is_file()
            }
            if stage == 'teacher':
                model = Teacher(native, recipe)
                sampler = StateSampler(list(rows), recipe)
            else:
                model = initialize_training_student(
                    native, recipe, config['initialization'], config.get('adapter')
                )
                if full:
                    enable_full_training(model, recipe)
                if config.get('method', 'pixel_kd_top20') == 'pixel_kd_top20':
                    cache = IndexedTeacherCache(config['teacher_cache'], config['teacher_index'])
                elif config['method'] != 'gt_top20':
                    raise ValueError('Public training supports Top20 with spatial KD or GT only')
            if full:
                local_world = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
                wrapped = wrap_full_training(
                    model,
                    processor,
                    device,
                    recipe,
                    {'nodes': world // local_world, 'processes': local_world},
                )
            else:
                wrapped = (
                    model if stage == 'teacher' else FullBatchStudent(model, processor, device)
                )
                if world > 1:
                    wrapped = torch.nn.parallel.DistributedDataParallel(
                        wrapped, device_ids=[local_rank] if device.type == 'cuda' else None
                    )
            groups = (
                parameter_groups(model, recipe) if full else adapter_parameter_groups(model, recipe)
            )
            optimizer = torch.optim.AdamW(groups, weight_decay=recipe['weight_decay'])
            identity = {
                'recipe': recipe,
                'stage': stage,
                'world_size': world,
                'steps': steps,
                'assets': asset_hashes,
                'data_sha256': file_sha256(config['data']),
                'base_model': config['base_model'],
                'initialization': config.get('initialization'),
                'method': config.get('method'),
                'adapter_sha256': file_sha256(config['adapter']) if config.get('adapter') else None,
                'teacher_sha256': file_sha256(config['teacher_cache']) if cache else None,
            }
            if resume and stage == 'teacher':
                previous = json.loads((Path(resume) / 'training.json').read_text())
                expected_rollout = previous['identity'].get('rollout_sha256')
                if expected_rollout:
                    source_rollout = Path(resume).parent / 'visited_states.json'
                    target_rollout = root / 'visited_states.json'
                    chosen = target_rollout if target_rollout.exists() else source_rollout
                    if not chosen.is_file() or file_sha256(chosen) != expected_rollout:
                        raise ValueError(
                            'Mixed-phase resume requires the original visited search states'
                        )
                    if rank == 0 and not target_rollout.exists():
                        shutil.copyfile(chosen, target_rollout)
                    if world > 1:
                        distributed.barrier()
                    identity['rollout_sha256'] = expected_rollout
                elif previous['step'] > recipe['supervised_steps']:
                    raise ValueError('Mixed-phase checkpoint is missing its visited-state identity')
            first = load_checkpoint(resume, wrapped, optimizer, identity) if resume else 0
            stream = SampleStream(len(rows), seed)
            rollout_ready = False
            for step in range(first, steps):
                if stage == 'teacher' and step >= recipe['supervised_steps'] and not rollout_ready:
                    rollout_file = root / 'visited_states.json'
                    if rank == 0 and not rollout_file.exists():
                        search = SearchTeacher(model, processor, recipe)
                        visited = {}
                        for index in range(min(len(rows), recipe['rollout_limit'])):
                            row = rows[index]
                            visited[row['id']] = visited_states(
                                row, search.predict(read_image(row), row['instruction'])
                            )
                        atomic_json(rollout_file, visited)
                    if world > 1:
                        distributed.barrier()
                    sampler = StateSampler(list(rows), recipe, json.loads(rollout_file.read_text()))
                    identity['rollout_sha256'] = file_sha256(rollout_file)
                    rollout_ready = True
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * schedule_rate(step, steps, recipe)
                accumulated = 0.0
                per_rank = batch // world
                for start in range(0, per_rank, micro):
                    positions = range(
                        step * batch + rank * per_rank + start,
                        step * batch + rank * per_rank + start + micro,
                    )
                    if stage == 'teacher':
                        prepared = []
                        for position in positions:
                            row, state = sampler.at(position)
                            prepared.append(
                                prepare_state(
                                    model,
                                    processor,
                                    read_image(row),
                                    row['instruction'],
                                    state,
                                    recipe,
                                )
                            )
                        loss = wrapped(
                            [value[0] for value in prepared],
                            [value[1] for value in prepared],
                            [value[2] for value in prepared],
                        )['loss']
                    else:
                        outputs, prepared = wrapped(
                            student_batch(model, rows, stream, positions, cache)
                        )
                        losses = [
                            unified_loss(
                                output,
                                value['valid'],
                                value['positive'],
                                value['target'],
                                value['layout'],
                                config.get('method', 'pixel_kd_top20'),
                                value['soft'],
                                recipe,
                            )[0]
                            for output, value in zip(outputs, prepared, strict=True)
                        ]
                        loss = torch.stack(losses).mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite training loss')
                    (loss * micro / per_rank).backward()
                    accumulated += float(loss.detach()) * micro / per_rank
                if full:
                    clip_full_gradients(wrapped, list(model.parameters()), recipe['gradient_clip'])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True
                    )
                optimizer.step()
                if rank == 0:
                    with (root / 'metrics.jsonl').open('a') as metrics:
                        metrics.write(
                            json.dumps({'step': step + 1, 'rank0_loss': accumulated}) + '\n'
                        )
                if (
                    (step + 1) % recipe['checkpoint_every'] == 0
                    or step + 1 == steps
                    or (stage == 'teacher' and step + 1 == recipe['supervised_steps'])
                ):
                    output = root / f'step_{step + 1:06d}'
                    save_checkpoint(output, wrapped, optimizer, step + 1, identity)
                    if not full and rank == 0:
                        from safetensors.torch import save_file

                        save_file(model.adapter_state(), str(output / 'adapter.safetensors'))
            return {'steps': steps, 'output': str(root)}
    finally:
        if cache:
            cache.close()
        if model:
            model.close()
        if world > 1 and distributed.is_initialized():
            distributed.destroy_process_group()
