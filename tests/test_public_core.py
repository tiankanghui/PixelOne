import json

import pytest
import torch
from PIL import Image

from pixelone.geometry import Layout
from pixelone.states import labels_for_roi
from pixelone.losses import support
from pixelone.losses import gt_objective
from pixelone.losses import top20_winner_loss
from pixelone.distillation import kd_loss
from pixelone.distillation import pixel_kd_loss
from pixelone.data import GroundingDataset, IndexedTeacherCache, index_teacher_cache, prepare_dataset
from pixelone.inference import decode_click
from pixelone.io import bundle_member
from pixelone.training import schedule_rate


@pytest.mark.parametrize('width,height', [(1, 1), (33, 65), (128, 97)])
def test_original_pixel_geometry(width, height):
    layout = Layout(width, height)
    assert int(layout.valid().sum()) == width * height
    assert layout.canvas_size == (32 * layout.cols, 32 * layout.rows)
    points = layout.positions()[layout.valid()]
    assert bool((points[:, 0] < width).all() and (points[:, 1] < height).all())


def test_decoder_commits_token_before_pixel_and_masks_padding():
    layout = Layout(33, 1)
    output = {'coarse': torch.tensor([1., 2.]), 'parents': torch.tensor([1, 0]),
              'fine': torch.zeros(2, 1024)}
    output['fine'][0, 1023] = 100.
    output['fine'][1, 0] = 1000.
    result = decode_click(output, layout)
    assert result['point'] == [32.5, .5]
    assert result['token_index'] == 1


@pytest.mark.parametrize('box,roi,expected', [([4, 4, 12, 12], [0, 0, 16, 16], 'inside'),
    ([4, 4, 12, 12], [16, 16, 32, 32], 'absent'), ([4, 4, 12, 12], [0, 0, 10, 16], 'truncated')])
def test_teacher_visibility_is_only_a_loss_label(box, roi, expected):
    labels = labels_for_roi(box, roi, 8)
    assert labels['visibility_status'] == expected
    assert labels['questions'][0]['enabled'] == (expected != 'truncated')
    if expected == 'truncated':
        assert all(question['enabled'] for question in labels['questions'][1:])


def test_top20_trains_selected_targets_and_non_target_competitors():
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 12).requires_grad_()
    valid = torch.ones_like(logits, dtype=torch.bool)
    positive = torch.zeros_like(valid)
    positive[:, :10] = True
    loss = top20_winner_loss(logits, valid, positive)
    expected = torch.stack([torch.log1p(torch.exp(logits[0, 10:] - logits[0, index]).sum())
                            for index in [9, 8]]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, :8]) == 0
    assert bool((logits.grad[0, 8:10] < 0).all() and (logits.grad[0, 10:] > 0).all())


def test_extra_kd_readouts_do_not_change_box_supervision():
    layout = Layout(128, 32)
    valid, positive = support(layout, [10, 0, 80, 32])
    target = layout.center_target([10, 0, 80, 32])
    coarse = torch.tensor([9., 2., 1., 4.])
    fine = torch.randn(4, 1024, requires_grad=True)
    outputs = [{'coarse': coarse, 'fine': fine[parents], 'parents': torch.tensor(parents)}
               for parents in [[1, 0], [1, 0, 2]]]
    losses = [gt_objective(output, valid, positive, target, layout, 1)[0] for output in outputs]
    torch.testing.assert_close(*losses, rtol=0, atol=0)


def test_vectorized_kd_preserves_observed_support_and_gradients():
    torch.manual_seed(9)
    coarse = torch.randn(3, requires_grad=True)
    fine = torch.randn(3, 4, requires_grad=True)
    output = {'coarse': coarse, 'fine': fine}
    soft = {'enabled': True, 'parents': torch.tensor([0, 2]),
            'mass': torch.tensor([[.1, .2, 0., 0.], [0., .3, .4, 0.]])}
    reference = kd_loss(output, soft, [0, 1, 2])[0]
    actual = pixel_kd_loss(output, soft, [0, 1, 2])[0]
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert coarse.grad[1] == 0 and bool((fine.grad[1] == 0).all())
    assert fine.grad[0, 2] == 0 and fine.grad[2, 3] == 0


def test_dataset_and_teacher_cache_bind_the_same_image_and_instruction(tmp_path):
    Image.new('RGB', (32, 32), 'blue').save(tmp_path / 'image.png')
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps({'id': 'example', 'image': 'image.png',
        'instruction': 'Click the blue button', 'bbox_xyxy': [4, 4, 12, 12]}) + '\n')
    target = tmp_path / 'train.jsonl'
    assert prepare_dataset(source, target)['targets'] == 1
    row = GroundingDataset(target)[0]
    cache = tmp_path / 'teacher.jsonl'
    cache.write_text(json.dumps({**row, 'source_image_sha256': row['image_sha256']}) + '\n')
    index = tmp_path / 'teacher.sqlite'
    index_teacher_cache(cache, index)
    reader = IndexedTeacherCache(cache, index)
    assert reader.get(row)['id'] == 'example'
    with pytest.raises(ValueError, match='identities differ'):
        reader.get({**row, 'instruction': 'different target'})
    reader.close()


@pytest.mark.parametrize('name', ['../secrets', '/etc/passwd', 'sub/../../file', '..\\file'])
def test_bundle_rejects_unsafe_member_paths(tmp_path, name):
    with pytest.raises(ValueError):
        bundle_member(tmp_path, name)


def test_learning_rate_schedule_bounds():
    recipe = {'lr_warmup_steps': 2, 'lr_schedule': 'cosine', 'min_lr_ratio': .1}
    assert schedule_rate(0, 10, recipe) == .5
    assert schedule_rate(1, 10, recipe) == 1.
    assert schedule_rate(9, 10, recipe) == pytest.approx(.1)
