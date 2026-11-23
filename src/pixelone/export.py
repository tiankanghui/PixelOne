"""Export a full student checkpoint as a checksummed inference model bundle."""

import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed.checkpoint as checkpoint
from safetensors.torch import save_file

from pixelone.io import atomic_json, bundle_member, file_sha256


def export_checkpoint(source, destination, *, precision='training_cuda', shard_bytes=2 * 1024**3):
    """Stream model-only shards from a complete Pro training checkpoint."""
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source / 'training.json').read_text())
    identity = metadata['identity']
    recipe = identity['recipe']
    if metadata.get('state') != 'complete' or recipe.get('train_mode') != 'full':
        raise ValueError('Export requires a complete full-parameter student checkpoint')
    if precision not in ('training_cuda', 'float32') or shard_bytes < 1:
        raise ValueError('Invalid export precision or shard budget')
    if destination.exists():
        raise ValueError('Choose a new model destination')
    assets = source.parent / 'assets'
    for name, expected in identity['assets'].items():
        if file_sha256(bundle_member(assets, name)) != expected:
            raise ValueError('Training processor/model configuration changed: ' + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    reader = checkpoint.FileSystemReader(source / 'state')
    entries = sorted(
        (name, item)
        for name, item in reader.read_metadata().state_dict_metadata.items()
        if name.startswith('model.student.')
    )
    if not entries or any('.lora_' in name for name, item in entries):
        raise ValueError('Expected merged full-student checkpoint tensors')
    files, weight_map, tensors, pending = {}, {}, {}, {}
    shard_number, pending_bytes, total_size = 0, 0, 0

    def flush():
        nonlocal shard_number, pending_bytes, total_size
        if not pending:
            return
        values = {
            name.removeprefix('model.'): torch.empty(item.size, dtype=item.properties.dtype)
            for name, item in pending.items()
        }
        checkpoint.load({'model': values}, storage_reader=reader, no_dist=True)
        exported = {}
        for name, value in values.items():
            name = name.removeprefix('student.')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite model tensor: ' + name)
            dtype = torch.float32
            if precision == 'training_cuda' and (
                name.startswith('native.backbone.language_model.layers.')
                or (recipe['train_vision'] and name.startswith('native.backbone.visual.blocks.'))
            ):
                dtype = torch.bfloat16
            exported[name] = value.to(dtype).contiguous()
            tensors[name] = {'shape': list(value.shape), 'dtype': str(dtype)}
            total_size += exported[name].numel() * exported[name].element_size()
        shard_number += 1
        filename = f'weights-{shard_number:05d}.safetensors'
        save_file(exported, str(staging / filename))
        files[filename] = file_sha256(staging / filename)
        weight_map.update({name: filename for name in exported})
        pending.clear()
        pending_bytes = 0

    for name, item in entries:
        if not isinstance(item, checkpoint.metadata.TensorStorageMetadata):
            raise ValueError('Expected tensor-only model state')
        size = (
            torch.Size(item.size).numel()
            * torch.empty((), dtype=item.properties.dtype).element_size()
        )
        if pending and pending_bytes + size > shard_bytes:
            flush()
        pending[name] = item
        pending_bytes += size
    flush()
    for name, expected in identity['assets'].items():
        shutil.copyfile(bundle_member(assets, name), staging / name)
        files[name] = expected
    manifest = {
        'schema': 'pixelone-pro-v1',
        'use_cache': False,
        'precision': precision,
        'recipe': recipe,
        'files': files,
        'weight_map': weight_map,
        'tensors': tensors,
        'source_step': metadata['step'],
        'source_checkpoint_sha256': file_sha256(source / 'training.json'),
        'runtime_package': 'pixelone-grounding==0.1.0',
    }
    atomic_json(staging / 'pixelone.json', manifest)
    atomic_json(
        staging / 'model.safetensors.index.json',
        {'metadata': {'total_size': total_size}, 'weight_map': weight_map},
    )
    os.rename(staging, destination)
    return {'destination': str(destination), 'shards': shard_number, 'tensor_bytes': total_size}
