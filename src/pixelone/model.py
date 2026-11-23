"""Shared heads, teacher/student wrappers, and adapter initialization."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Low-rank updates around a frozen projection, using FP32 adapter parameters."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        value = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return value + self.scale * delta.to(value.dtype)


def install_lora(native, rank, alpha):
    targets = {
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'in_proj_qkv',
        'in_proj_z',
        'in_proj_b',
        'in_proj_a',
        'out_proj',
    }
    language = native.backbone.language_model
    names = []
    for name, module in list(language.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in targets:
            parent, leaf = name.rsplit('.', 1)
            language.get_submodule(parent).__setattr__(leaf, LoRALinear(module, rank, alpha))
            names.append(name)
    for index, layer in enumerate(language.layers):
        selected = [n.rsplit('.', 1)[-1] for n in names if n.startswith(f'layers.{index}.')]
        expected = (
            {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
            if layer.layer_type == 'full_attention'
            else targets - {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
        )
        if set(selected) != expected:
            raise ValueError(
                f'Incomplete LoRA coverage in {layer.layer_type} layer {index}: {selected}'
            )
    return names


def tensor_hash(values):
    sha = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().float().cpu().contiguous()
        sha.update(name.encode())
        sha.update(str(list(value.shape)).encode())
        sha.update(value.numpy().tobytes())
    return sha.hexdigest()


def choice_loss(logits, mask, labels):
    if logits.shape != mask.shape or labels.shape != (logits.shape[0],):
        raise ValueError('Choice loss shapes differ')
    active = labels != -100
    if not active.any():
        raise ValueError('Training state has no unambiguous supervised question')
    if not torch.isfinite(logits[mask]).all():
        raise ValueError('Nonfinite native JEV logits')
    chosen = labels[active]
    if (
        (chosen < 0).any()
        or (chosen >= logits.shape[1]).any()
        or not mask[active].gather(1, chosen[:, None]).all()
    ):
        raise ValueError('Target refers to an invalid option')
    return F.cross_entropy(logits.float()[active].masked_fill(~mask[active], -torch.inf), chosen)


class Teacher(nn.Module):
    """Native decision model with language LoRA and a trainable query/key head."""

    def __init__(self, native, recipe):
        super().__init__()
        self.native = native
        self.recipe = recipe
        from pixelone.processor import legacy_unpadded_mask

        native.prepare_for_inference()
        self.initial_head_sha256 = tensor_hash(native.head.state_dict())
        ids = [native.config.decide_token_id, native.config.option_end_token_id]
        embedding = native.get_input_embeddings()
        self.initial_special_embeddings_sha256 = tensor_hash({'special': embedding.weight[ids]})
        self.lora_names = install_lora(native, recipe['lora_rank'], recipe['lora_alpha'])
        native.head.requires_grad_(True)
        # A shared additive bias cancels in categorical CE; preserve its grounding calibration.
        native.head.bias.requires_grad_(False)
        if recipe.get('gradient_checkpointing', True):
            native.backbone.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        self.mask_hook = native.backbone.language_model.register_forward_pre_hook(
            legacy_unpadded_mask, with_kwargs=True
        )

    def train(self, mode=True):
        super().train(mode)
        if not self.recipe.get('train_vision', False):
            self.native.backbone.visual.eval()
        return self

    def forward(self, group, features, labels=None):
        if isinstance(group, (list, tuple)):
            from .batching import forward_states

            return forward_states(self.native, group, features, labels)
        if features.is_inference():
            raise ValueError('Clone frozen visual features outside inference_mode before backward')
        output = self.native(**group, visual_features=features)
        result = dict(
            logits=output.logits, probabilities=output.probabilities, score_mask=output.score_mask
        )
        if labels is not None:
            result['loss'] = choice_loss(
                output.logits, output.score_mask, labels.to(output.logits.device)
            )
        return result

    def adapter_state(self):
        return {
            n: p.detach().cpu().contiguous()
            for n, p in self.named_parameters()
            if '.lora_' in n or n.startswith('native.head.')
        }

    def load_adapter(self, state):
        expected = {
            n: p
            for n, p in self.named_parameters()
            if '.lora_' in n or n.startswith('native.head.')
        }
        if set(state) != set(expected):
            raise ValueError('Adapter keys differ from native JEV head/LoRA')
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != expected[name].shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor')
                expected[name].copy_(value.to(expected[name]))

    def close(self):
        self.mask_hook.remove()


class FineReadout(nn.Module):
    """Query-conditioned pixel logits from a depthwise-mixed visual token grid."""

    def __init__(self, hidden, width=128, subdivisions=32):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.visual = nn.Linear(hidden, width)
        self.query = nn.Linear(hidden, 2 * width)
        self.local = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.mix = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, subdivisions**2)
        )

    def forward(self, spatial, query, layout, parents=None):
        """Mix the full grid, then apply the MLP to the requested tokens."""
        if spatial.shape[0] != layout.count:
            raise ValueError('Lost visual spatial layout')
        v = self.visual(self.norm(spatial.float()))
        scale, bias = self.query(self.norm(query.float())).chunk(2, -1)
        v = v * (1 + scale.tanh()) + bias
        image = v.T.reshape(1, -1, layout.rows, layout.cols)
        v = v + self.local(image).reshape(v.shape[1], -1).T
        if parents is not None:
            v = v[parents]
        return self.mix(v)


def route(coarse, required=None, positive=None, topk=4):
    # Inference uses only coarse logits. Training adds GT/KD support, without
    # inserting GT into the image, text or language-model attention.
    """Use one winning token at inference and the GT/prediction union during training."""
    if required is None:
        return coarse.detach().argmax().reshape(1)
    values = list(required)
    values += coarse.detach().topk(min(topk, len(coarse))).indices.tolist()
    if positive is not None:
        pos = positive.to(coarse.device).nonzero().flatten()
        if not len(pos):
            raise ValueError('No valid GT regions to route')
        values += pos[coarse.detach()[pos].topk(min(topk, len(pos))).indices].tolist()
    return torch.tensor(list(dict.fromkeys(values)), device=coarse.device, dtype=torch.long)


class ClickStudent(Teacher):
    """Native decision model with a pixel readout and language LoRA."""

    def __init__(self, native, recipe):
        super().__init__(native, recipe)
        self.spatial = FineReadout(
            native.head.q.in_features,
            recipe.get('readout_width', 128),
            recipe.get('subdivisions', 32),
        ).to(device=native.head.q.weight.device, dtype=torch.float32)

    def forward(self, groups, features, layouts, required=None, positive=None, topk=4):
        if not isinstance(groups, (list, tuple)):
            raise ValueError('Provide a batch, including for one screenshot')
        count = len(groups)
        if (
            not count
            or len(features) != count
            or len(layouts) != count
            or any(len(group['query_indices']) != 1 for group in groups)
        ):
            raise ValueError('Each batch item requires one screenshot and one query')
        required = [None] * count if required is None else required
        positive = [None] * count if positive is None else positive
        if len(required) != count or len(positive) != count:
            raise ValueError('Batched routes differ from targets')
        captured = []
        hook = self.native.head.register_forward_pre_hook(
            lambda module, args: captured.append(args)
        )
        try:
            Teacher.forward(self, groups, features)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError('Expected one batched native readout')
        pairs, queries = captured[0]
        if pairs.shape[0] != count or queries.shape[0] != count:
            raise ValueError('Lost batched sample identities')
        result = []
        for index, layout in enumerate(layouts):
            spatial, query = pairs[index, : layout.count], queries[index]
            if len(features[index]) != layout.count:
                raise ValueError('Visual grid differs from spatial layout')
            coarse = self.native.head(spatial[None], query[None])[0]
            parents = route(coarse, required[index], positive[index], topk)
            result.append(
                dict(
                    coarse=coarse,
                    fine=self.spatial(spatial, query, layout, parents),
                    parents=parents,
                )
            )
        return result

    def adapter_state(self):
        return {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in self.named_parameters()
            if '.lora_' in name or name.startswith(('native.head.', 'spatial.'))
        }

    def load_adapter(self, state):
        if not any(name.startswith('spatial.') for name in state):
            return super().load_adapter(state)
        expected = {
            name: parameter
            for name, parameter in self.named_parameters()
            if '.lora_' in name or name.startswith(('native.head.', 'spatial.'))
        }
        if expected.keys() != state.keys():
            raise ValueError('Spatial student adapter keys differ')
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != expected[name].shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor')
                expected[name].copy_(value.to(expected[name]))


class FullWeightStudent(ClickStudent):
    """Inference wrapper for a merged full-parameter student without LoRA modules."""

    def __init__(self, native, recipe):
        torch.nn.Module.__init__(self)
        self.native, self.recipe = native, recipe
        self.native.head.bias = torch.nn.Parameter(torch.zeros(1))
        self.spatial = FineReadout(
            native.head.q.in_features, recipe['readout_width'], recipe['subdivisions']
        )
        self.native.full_parameter_training = True
        self.native.train_vision = recipe['train_vision']

    def forward(self, groups, features, layouts, required=None, positive=None, topk=4):
        if not isinstance(groups, (tuple, list)):
            raise ValueError('Full inference always uses the same batched path as training')
        return super().forward(groups, features, layouts, required, positive, topk)

    def close(self):
        pass


def fresh_head(hidden, recipe):
    """Initialize the pixel head on CPU without changing the surrounding RNG state."""
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.random.default_generator.manual_seed(recipe['head_seed'])
        return FineReadout(hidden, recipe['readout_width'], recipe['subdivisions'])


def initialize_student(native, recipe, teacher_state):
    """Copy the teacher adapter and create a separately seeded pixel head."""
    if any(n.startswith('spatial.') for n in teacher_state):
        raise ValueError('Initialization requires T3, not an already trained S1 checkpoint')
    model = ClickStudent(native, recipe)
    model.load_adapter(teacher_state)
    head = fresh_head(native.head.q.in_features, recipe)
    model.spatial.load_state_dict(head.state_dict())
    state = model.adapter_state()
    shared = {n: v for n, v in state.items() if not n.startswith('spatial.')}
    if tensor_hash(shared) != tensor_hash(teacher_state):
        raise ValueError('Teacher shared initialization changed')
    expected = {
        n
        for n, p in model.named_parameters()
        if '.lora_' in n or n.startswith(('native.head.', 'spatial.'))
    }
    for n, p in model.named_parameters():
        should_train = n in expected and n != 'native.head.bias'
        if p.requires_grad != should_train:
            raise ValueError('Unexpected frozen/trainable parameter: ' + n)
    initial = dict(
        shared_sha256=tensor_hash(shared),
        spatial_sha256=tensor_hash(head.state_dict()),
        adapter_tensor_sha256=tensor_hash(state),
        head_seed=recipe['head_seed'],
        scope='fresh T3 shared adapter plus fixed CPU-seeded fine head; no prior S1 checkpoint',
    )
    model.unified_initialization = initial
    return model, initial


def initialize_continued_student(native, recipe, state, expected):
    """Retain the trained pixel head; this is not a fresh-head teacher initialization."""
    if not any(n.startswith('spatial.') for n in state):
        raise ValueError('Continuation requires a trained pixel head')
    model = ClickStudent(native, recipe)
    model.load_adapter(state)
    loaded = model.adapter_state()
    if tensor_hash(loaded) != expected['adapter_tensor_sha256'] or tensor_hash(
        loaded
    ) != tensor_hash(state):
        raise ValueError('Continued student weights were not loaded exactly')
    fine = {k.removeprefix('spatial.'): v for k, v in loaded.items() if k.startswith('spatial.')}
    if tensor_hash(fine) != expected['spatial_sha256']:
        raise ValueError('Trained pixel head was reset')
    model.unified_initialization = dict(expected)
    return model, model.unified_initialization
