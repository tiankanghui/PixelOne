"""Save and resume model, optimizer, RNG, and training-identity state."""

from pathlib import Path

import torch
import torch.distributed as distributed
import torch.distributed.checkpoint as checkpoint
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from pixelone.io import atomic_json, stable_hash


def initialized():
    return distributed.is_available() and distributed.is_initialized()


def save_checkpoint(directory, model, optimizer, step, identity):
    """Save training state and RNG state, marking metadata complete only after saving."""
    directory = Path(directory)
    rank = distributed.get_rank() if initialized() else 0
    if rank == 0:
        if directory.exists():
            raise ValueError('Checkpoint destination already exists')
        directory.mkdir(parents=True)
    if initialized():
        distributed.barrier()
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=StateDictOptions(cpu_offload=True)
    )
    checkpoint.save(
        {'model': model_state, 'optimizer': optimizer_state}, checkpoint_id=directory / 'state'
    )
    random_state = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        random_state['cuda'] = torch.cuda.get_rng_state()
    torch.save(random_state, directory / f'rng_rank_{rank:04d}.pt')
    if initialized():
        distributed.barrier()
    if rank == 0:
        atomic_json(
            directory / 'training.json',
            {
                'schema': 'pixelone-training-v1',
                'step': step,
                'identity': identity,
                'identity_sha256': stable_hash(identity),
                'state': 'complete',
            },
        )


def load_checkpoint(directory, model, optimizer, identity):
    """Validate run identity, restore training state, and return the next step index."""
    import json

    directory = Path(directory)
    metadata = json.loads((directory / 'training.json').read_text())
    if metadata.get('state') != 'complete' or metadata['identity_sha256'] != stable_hash(identity):
        raise ValueError(
            'Checkpoint is incomplete or belongs to another training recipe/data/world size'
        )
    options = StateDictOptions(cpu_offload=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    state = {'model': model_state, 'optimizer': optimizer_state}
    checkpoint.load(state, checkpoint_id=directory / 'state')
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state['model'],
        optim_state_dict=state['optimizer'],
        options=options,
    )
    rank = distributed.get_rank() if initialized() else 0
    random_state = torch.load(
        directory / f'rng_rank_{rank:04d}.pt', map_location='cpu', weights_only=True
    )
    torch.set_rng_state(random_state['cpu'])
    if 'cuda' in random_state and torch.cuda.is_available():
        torch.cuda.set_rng_state(random_state['cuda'])
    return metadata['step']
