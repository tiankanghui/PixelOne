import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from pixelone.distillation import kd_loss, pixel_kd_loss
from pixelone.losses import unified_loss
from pixelone.runtime import OPTIONAL_KERNELS, full_layer_attention, inference_precision, qwen_kernel_backend


def test_cli_retains_the_core_workflows():
    result = subprocess.run([sys.executable, '-m', 'pixelone.cli', '--help'],
        env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for command in ('infer', 'train', 'prepare-data', 'cache-teacher', 'export'):
        assert command in result.stdout


@pytest.mark.parametrize('method', ['gt', 'gt_set', 'pixel_kd', 'region_kd', 'hard_kd', 'pixel_kd_top100'])
def test_only_supported_training_objectives_are_exposed(method):
    with pytest.raises(ValueError, match='Top20'):
        unified_loss(None, None, None, None, None, method, None, {})


def test_code_has_no_historical_package_dependencies():
    root = Path(__file__).resolve().parents[1] / 'src'
    obsolete = {'jev_click_finetune', 'jev_fine_ablation', 'jev_fine_refine', 'jev_spatial_distill',
                'jev_unified_student', 'qwen_spatial', 'spatial_pro_experiments', 'scripts'}
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert all(name.split('.')[0] not in obsolete for name in names), path


@pytest.mark.parametrize('enabled', [False, True])
def test_kd_reference_and_vectorized_gradients_match(enabled):
    torch.manual_seed(31)
    coarse = torch.randn(4)
    fine = torch.randn(3, 8)
    soft = dict(enabled=enabled, parents=torch.tensor([2, 0]), mass=torch.tensor([
        [0., .2, .1, 0., 0., .1, 0., 0.], [.3, 0., 0., 0., .1, .2, 0., 0.]]))
    results = []
    for implementation in (kd_loss, pixel_kd_loss):
        output = dict(coarse=coarse.clone().requires_grad_(), fine=fine.clone().requires_grad_())
        loss, terms = implementation(output, soft, [0, 3, 2])
        loss.backward()
        results.append((loss.detach(), output['coarse'].grad, output['fine'].grad, terms))
    for reference, actual in zip(results[0][:3], results[1][:3], strict=True):
        torch.testing.assert_close(reference, actual)
    assert results[0][3] == results[1][3]


def test_kernel_and_attention_contexts_restore_on_failure():
    original_mask = object()
    originals = {name: object() for name in OPTIONAL_KERNELS}
    module = SimpleNamespace(**originals, is_fast_path_available=True, create_causal_mask=original_mask)
    with pytest.raises(RuntimeError, match='deliberate'):
        with full_layer_attention(module, 'bidirectional_full_layers'), qwen_kernel_backend('torch', module):
            assert all(getattr(module, name) is None for name in OPTIONAL_KERNELS)
            embeds = torch.ones(2, 3, 4)
            padding = torch.tensor([[1, 1, 1], [1, 0, 0]])
            mask = module.create_causal_mask(SimpleNamespace(_attn_implementation='sdpa'), embeds, padding)
            assert mask[0].eq(0).all()
            assert mask[1, 0, 0, 1:].eq(torch.finfo(embeds.dtype).min).all()
            raise RuntimeError('deliberate')
    assert module.create_causal_mask is original_mask
    assert module.is_fast_path_available
    assert all(getattr(module, name) is value for name, value in originals.items())


def test_inference_precision_preserves_fp32_parameters():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer('floating', torch.ones(2))
            self.register_buffer('integer', torch.ones(2, dtype=torch.int64))

        def forward(self, hidden_states):
            return hidden_states

    language, vision = Block(), Block()
    head = torch.nn.Linear(2, 2)
    model = SimpleNamespace(native=SimpleNamespace(head=head, backbone=SimpleNamespace(
        language_model=SimpleNamespace(layers=[language]), visual=SimpleNamespace(blocks=[vision]))))
    inference_precision(model, dict(train_vision=True), 'training_cuda')
    for block in (language, vision):
        assert block.weight.dtype == torch.float32
        assert block.floating.dtype == torch.bfloat16
        assert block.integer.dtype == torch.int64
        assert block(torch.ones(2)).dtype == torch.bfloat16
    assert head.weight.dtype == torch.float32
