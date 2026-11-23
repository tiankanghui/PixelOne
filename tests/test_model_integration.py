import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from conftest import tiny_native
from pixelone.model import Teacher
from pixelone.runtime import native_runtime
from pixelone.model import FullWeightStudent
from pixelone.inference import PixelOnePredictor
from pixelone.io import file_sha256
from pixelone.teacher import SearchTeacher
from pixelone.training import initialize_training_student


def recipe():
    return dict(lora_rank=2, lora_alpha=4, lora_lr=.001, head_lr=.001, spatial_lr=.001,
        full_lr=.001, vision_lr=.001, weight_decay=.01, gradient_clip=1., global_batch=1,
        micro_batch=1, seed=31, head_seed=31, readout_width=16, subdivisions=32,
        routing_topk=2, gradient_checkpointing=False, checkpoint_every=1, kd_weight=.5,
        temperature=2., decoder='coarse_first', prompt='valid_click_v1', kernel_backend='torch',
        attention='sdpa', attention_mode='causal', lr_schedule='constant', sharding='full_shard')


@pytest.mark.parametrize('moe', [False, True])
def test_t3_really_uses_three_language_calls_and_five_visual_encodings(processor, moe):
    configuration = {**recipe(), 'full_view_size': 128, 'crop_view_size': 128}
    with native_runtime():
        teacher = Teacher(tiny_native(processor, moe), configuration)
        counts = {'language': 0, 'vision': 0}
        def language_hook(*arguments):
            counts['language'] += 1
        def vision_hook(*arguments):
            counts['vision'] += 1
        hooks = [teacher.native.backbone.language_model.register_forward_hook(language_hook),
                 teacher.native.backbone.visual.register_forward_hook(vision_hook)]
        try:
            prediction = SearchTeacher(teacher, processor, configuration).predict(
                Image.new('RGB', (64, 64), 'blue'), 'Click the blue area')
            assert len(prediction['stages']) == 3
            assert counts == {'language': 3, 'vision': 5}
            assert all(0 <= value <= 64 for value in prediction['predicted_xy'])
        finally:
            for hook in hooks:
                hook.remove()
            teacher.close()


@pytest.mark.parametrize('moe', [False, True])
def test_public_full_weight_load_reproduces_direct_logits(processor, tmp_path, moe):
    configuration = {**recipe(), 'train_mode': 'full', 'train_vision': True,
                     'attention_mode': 'bidirectional_full_layers'}
    with native_runtime(attention_mode=configuration['attention_mode']):
        model = FullWeightStudent(tiny_native(processor, moe), configuration).eval()
        model.native.config.to_json_file(tmp_path / 'config.json', use_diff=False)
        state = {name: value.contiguous() for name, value in model.state_dict().items()}
        save_file(state, str(tmp_path / 'weights.safetensors'))
        for filename in ['processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
            shutil.copyfile(Path(os.environ['PIXELONE_TEST_PROCESSOR']) / filename, tmp_path / filename)
        files = {path.name: file_sha256(path) for path in tmp_path.iterdir()}
        manifest = dict(schema='pixelone-pro-v1', use_cache=False, recipe=configuration, precision='float32',
            files=files, weight_map={name: 'weights.safetensors' for name in state},
            tensors={name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in state.items()})
        (tmp_path / 'pixelone.json').write_text(json.dumps(manifest))
        original = PixelOnePredictor(model, processor, manifest, 'cpu')
        restored = PixelOnePredictor.from_pretrained(tmp_path, device='cpu')
        image = Image.new('RGB', (65, 33), 'blue')
        before = original.predict_outputs([image], ['Click the blue area'])[0][0]
        after = restored.predict_outputs([image], ['Click the blue area'])[0][0]
        for key in ['coarse', 'fine', 'parents']:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        assert restored.predict(image, 'Click the blue area')['original_size'] == [65, 33]
        image.save(tmp_path / 'screenshot.png')
        result = subprocess.run([sys.executable, '-m', 'pixelone.cli', 'infer', '--model', str(tmp_path),
            '--image', str(tmp_path / 'screenshot.png'), '--instruction', 'Click the blue area', '--device', 'cpu'],
            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == restored.predict(image, 'Click the blue area')


@pytest.mark.parametrize('full', [False, True])
def test_training_checkpoint_resume_matches_uninterrupted_student(processor, tmp_path, monkeypatch, full):
    from pixelone import training
    configuration = recipe()
    if full:
        configuration.update(train_mode='full', train_vision=True, attention_mode='bidirectional_full_layers')
    image = tmp_path / 'image.png'
    Image.new('RGB', (32, 32), 'blue').save(image)
    data = tmp_path / 'train.jsonl'
    data.write_text(json.dumps({'id': 'blue', 'image': 'image.png',
        'instruction': 'Click blue', 'bbox_xyxy': [4, 4, 24, 24]}) + '\n')
    native = None
    with native_runtime(attention_mode=configuration['attention_mode']):
        native = tiny_native(processor)
        monkeypatch.setattr(training, 'load_native', lambda path, device: (copy.deepcopy(native), processor))
        adapter = tmp_path / 'initial.safetensors'
        student = initialize_training_student(copy.deepcopy(native), configuration, 'base')
        save_file(student.adapter_state(), str(adapter))
        student.close()
    config = {'stage': 'student', 'recipe': configuration, 'steps': 2, 'data': str(data),
        'image_root': str(tmp_path), 'output': str(tmp_path / 'original'), 'base_model': 'tiny',
        'device': 'cpu', 'initialization': 'student', 'adapter': str(adapter), 'method': 'gt_top20'}
    training.run_training(config)
    resumed = {**config, 'output': str(tmp_path / 'resumed')}
    training.run_training(resumed, resume=tmp_path / 'original/step_000001')
    first = json.loads((tmp_path / 'original/metrics.jsonl').read_text().splitlines()[-1])
    second = json.loads((tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()[-1])
    assert first == second
    if full:
        from pixelone.export import export_checkpoint
        exported = tmp_path / 'exported'
        export_checkpoint(tmp_path / 'original/step_000002', exported, precision='float32', shard_bytes=4 * 1024**2)
        loaded = PixelOnePredictor.from_pretrained(exported, device='cpu')
        assert len(loaded.predict(Image.new('RGB', (32, 32), 'blue'), 'Click blue')['point']) == 2


def test_teacher_mixed_phase_resume_reuses_original_visited_states(processor, tmp_path, monkeypatch):
    from pixelone import training
    configuration = {**recipe(), 'supervised_steps': 1, 'mixed_steps': 2,
        'full_view_size': 128, 'crop_view_size': 128, 'rollout_limit': 1, 'rollout_fraction': 1.}
    Image.new('RGB', (32, 32), 'blue').save(tmp_path / 'image.png')
    data = tmp_path / 'teacher_data.jsonl'
    data.write_text(json.dumps({'id': 'blue', 'image': 'image.png',
        'instruction': 'Click blue', 'bbox_xyxy': [4, 4, 24, 24]}) + '\n')
    with native_runtime():
        native = tiny_native(processor)
    monkeypatch.setattr(training, 'load_native', lambda path, device: (copy.deepcopy(native), processor))
    config = {'stage': 'teacher', 'recipe': configuration, 'data': str(data),
        'image_root': str(tmp_path), 'output': str(tmp_path / 'original'), 'base_model': 'tiny', 'device': 'cpu'}
    training.run_training(config)
    resumed = {**config, 'output': str(tmp_path / 'resumed')}
    training.run_training(resumed, resume=tmp_path / 'original/step_000002')
    assert (tmp_path / 'original/visited_states.json').read_bytes() == (tmp_path / 'resumed/visited_states.json').read_bytes()
    assert (tmp_path / 'original/step_000003/adapter.safetensors').read_bytes() == (
        tmp_path / 'resumed/step_000003/adapter.safetensors').read_bytes()
