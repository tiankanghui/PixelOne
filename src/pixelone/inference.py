"""Load local or Hub model bundles and decode token-first pixel predictions."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading

import torch

from pixelone.inputs import prepare
from pixelone.runtime import native_runtime
from pixelone.model import FullWeightStudent
from pixelone.runtime import inference_precision
from pixelone.io import bundle_member, file_sha256

RUNTIME_LOCK = threading.RLock()
BUNDLE_SCHEMA = 'pixelone-pro-v1'


def resolve_model_path(model_path, *, subfolder=None, revision=None):
    """Resolve local weights or fetch only the selected Hub model subfolder."""
    folder = str(subfolder) if subfolder else ''
    if folder:
        folder = str(bundle_member(Path('.'), folder))
        if folder == '.':
            raise ValueError('Select a model subfolder, not the current directory')
    root = Path(model_path).expanduser()
    if root.is_dir():
        selected = root / folder
        if not selected.is_dir():
            raise FileNotFoundError('Model subfolder not found: ' + str(selected))
        return selected
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError

    prefix = folder.rstrip('/') + '/' if folder else ''
    try:
        manifest_path = hf_hub_download(
            str(model_path), filename=prefix + 'pixelone.json', revision=revision
        )
    except EntryNotFoundError as error:
        raise ValueError(
            'No PixelOne manifest at this location; select a model subfolder for a collection repository'
        ) from error
    manifest = json.loads(Path(manifest_path).read_text())
    members = list(manifest['files'])
    for name in members:
        bundle_member(Path('.'), name)
    root = Path(
        snapshot_download(
            str(model_path),
            revision=revision,
            allow_patterns=[prefix + 'pixelone.json', *[prefix + name for name in members]],
        )
    )
    return root / folder


def inspect_bundle(root, verify_weights=True):
    root = Path(root)
    manifest = json.loads((root / 'pixelone.json').read_text())
    if manifest.get('schema') != BUNDLE_SCHEMA or manifest.get('use_cache') is not False:
        raise ValueError('Expected a PixelOne Pro bundle with KV caching disabled')
    recipe = manifest['recipe']
    if (
        recipe.get('train_mode') != 'full'
        or recipe.get('decoder') != 'coarse_first'
        or recipe.get('subdivisions') != 32
        or recipe.get('prompt') != 'valid_click_v1'
        or recipe.get('attention_mode') != 'bidirectional_full_layers'
        or recipe.get('train_vision') is not True
    ):
        raise ValueError('Unsupported Pro readout, input, or attention contract')
    if recipe.get('kernel_backend') not in ('torch', 'native'):
        raise ValueError('Unknown kernel backend')
    if manifest.get('precision') not in ('training_cuda', 'float32'):
        raise ValueError('Unknown tensor precision contract')
    weights = manifest['weight_map']
    if not weights or set(weights) != set(manifest['tensors']):
        raise ValueError('Incomplete tensor index')
    if not set(weights.values()) <= manifest['files'].keys():
        raise ValueError('Weight index contains an unhashed shard')
    for name, expected in manifest['files'].items():
        path = bundle_member(root, name)
        if not path.is_file():
            raise FileNotFoundError(path)
        if (verify_weights or not name.endswith('.safetensors')) and file_sha256(path) != expected:
            raise ValueError('Bundle checksum mismatch: ' + name)
    return manifest


def decode_click(output, layout):
    """Select one token, then its highest-scoring non-padding pixel."""
    parent = int(output['coarse'].detach().argmax())
    parents = output['parents'].tolist()
    if parent not in parents:
        raise ValueError('Selected token was not read by the pixel head')
    row = parents.index(parent)
    logits = output['fine'][row].float()
    valid = layout.valid(logits.device)[parent]
    if not bool(valid.any()):
        raise ValueError('Selected token contains no image pixel')
    pixel = int(logits.masked_fill(~valid, -torch.inf).argmax())
    point = layout.positions(logits.device)[parent, pixel].tolist()
    return {'point': point, 'token_index': parent, 'pixel_index': pixel}


class PixelOnePredictor:
    """Reusable screenshot-to-click interface for a loaded Pro student."""

    def __init__(self, model, processor, manifest, device, kernel_backend=None):
        self.model, self.processor, self.manifest = model, processor, manifest
        self.device = torch.device(device)
        self.recipe = dict(manifest['recipe'])
        self.kernel_backend = kernel_backend or self.recipe['kernel_backend']

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        device='cuda:0',
        revision=None,
        subfolder=None,
        verify_weights=True,
        kernel_backend=None,
    ):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from pixelone.processor import LegacyProcessorAdapter
        from videojev_hf.configuration_videojev import VideoJevConfig
        from videojev_hf.modeling_videojev import VideoJevForVideoUnderstanding

        root = resolve_model_path(model_path, subfolder=subfolder, revision=revision)
        manifest = inspect_bundle(root, verify_weights)
        device = torch.device(device)
        if manifest['precision'] == 'training_cuda' and device.type != 'cuda':
            raise ValueError('Released mixed-precision Pro weights require CUDA')
        recipe = manifest['recipe']
        backend = kernel_backend or recipe['kernel_backend']
        with RUNTIME_LOCK, native_runtime(backend, attention_mode=recipe['attention_mode']):
            config = VideoJevConfig.from_dict(json.loads((root / 'config.json').read_text()))
            config._attn_implementation = recipe.get('attention', 'sdpa')
            with init_empty_weights(include_buffers=False):
                model = FullWeightStudent(VideoJevForVideoUnderstanding(config), recipe)
            expected = model.state_dict()
            if set(expected) != set(manifest['weight_map']):
                raise ValueError('Model structure differs from the weight index')
            loaded = set()
            for filename in sorted(set(manifest['weight_map'].values())):
                values = load_file(str(bundle_member(root, filename)), device='cpu')
                for name, tensor in values.items():
                    specification = manifest['tensors'].get(name)
                    if (
                        name in loaded
                        or manifest['weight_map'].get(name) != filename
                        or specification is None
                        or tensor.shape != expected[name].shape
                        or list(tensor.shape) != specification['shape']
                        or str(tensor.dtype) != specification['dtype']
                    ):
                        raise ValueError('Tensor shape, dtype or shard mismatch: ' + name)
                    loaded.add(name)
                model.load_state_dict(
                    {name: tensor.to(device) for name, tensor in values.items()},
                    strict=False,
                    assign=True,
                )
                del values
            if loaded != set(expected) or any(
                parameter.is_meta for parameter in model.parameters()
            ):
                raise ValueError('Some model parameters were not loaded')
            model.to(device).eval().requires_grad_(False)
            inference_precision(model, recipe, manifest['precision'])
            if recipe.get('experts_backend') == 'grouped_mm' and model.native.is_moe:
                for module in model.native.backbone.language_model.modules():
                    if type(module).__name__ == 'Qwen3_5MoeExperts':
                        module.config._experts_implementation = 'grouped_mm'
            processor = LegacyProcessorAdapter(
                AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
            )
            for token, expected_id in [
                ('<|decide|>', config.decide_token_id),
                ('<|option_end|>', config.option_end_token_id),
            ]:
                if processor.tokenizer.convert_tokens_to_ids(token) != expected_id:
                    raise ValueError('Native decision token identity differs')
        return cls(model, processor, manifest, device, backend)

    @contextmanager
    def activated(self):
        with (
            RUNTIME_LOCK,
            native_runtime(self.kernel_backend, attention_mode=self.recipe['attention_mode']),
            torch.no_grad(),
        ):
            yield

    def predict_outputs(self, images, instructions):
        if not images or len(images) != len(instructions):
            raise ValueError('Each screenshot requires one instruction')
        with self.activated():
            prepared = [
                prepare(self.model, self.processor, image, instruction)
                for image, instruction in zip(images, instructions, strict=True)
            ]
            outputs = self.model(
                [value[0] for value in prepared],
                [value[1] for value in prepared],
                [value[2] for value in prepared],
            )
        return outputs, [value[2] for value in prepared], [value[3] for value in prepared]

    def predict(self, image, instruction):
        outputs, layouts, metadata = self.predict_outputs([image.convert('RGB')], [instruction])
        return {
            **decode_click(outputs[0], layouts[0]),
            'image_size': list(image.size),
            'kernel_backend': self.kernel_backend,
            **metadata[0],
        }
