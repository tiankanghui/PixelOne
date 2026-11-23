"""Construct and sample box-derived or teacher-visited training states."""

import hashlib
import io
import math
import random
from pathlib import Path

from PIL import Image

from pixelone.sampling import SampleStream
from pixelone.io import stable_hash


def xyxy(values):
    if len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Expected four finite xyxy coordinates')
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError('Degenerate box/ROI')
    return [float(v) for v in values]


def labels_for_roi(box, roi, cells, *, local=True):
    """Match CENTER questions and conservatively label geometric visibility.

    Truncated boxes mask visibility: intersection alone does not establish
    recognizability. Location is supervised only when the center is in the ROI.
    """
    if cells not in (4, 8) or type(cells) is not int:
        raise ValueError('Current T3 uses 4/8 grids')
    box, roi = xyxy(box), xyxy(roi)
    overlap = [max(0.0, min(box[i + 2], roi[i + 2]) - max(box[i], roi[i])) for i in (0, 1)]
    overlap_area = overlap[0] * overlap[1]
    coverage = overlap_area / ((box[2] - box[0]) * (box[3] - box[1]))
    contained = all(roi[i] <= box[i] and box[i + 2] <= roi[i + 2] for i in (0, 1))
    visibility = 'inside' if contained else 'absent' if overlap_area == 0 else 'truncated'
    center = [(box[i] + box[i + 2]) / 2 for i in (0, 1)]
    center_inside = all(roi[i] <= center[i] < roi[i + 2] for i in (0, 1))
    normalized = [(center[i] - roi[i]) / (roi[i + 2] - roi[i]) for i in (0, 1)]
    indices = (
        [min(cells - 1, math.floor(v * cells)) for v in normalized]
        if center_inside
        else [None, None]
    )
    questions = []
    if local:
        questions.append(
            dict(
                id='presence',
                enabled=visibility != 'truncated',
                label_index=0 if visibility == 'inside' else 1 if visibility == 'absent' else None,
            )
        )
    questions.extend(
        dict(id=name, enabled=center_inside, label_index=index)
        for name, index in zip(('horizontal', 'vertical'), indices)
    )
    return dict(
        cells_per_axis=cells,
        visibility_status=visibility,
        box_coverage=coverage,
        center_inside_roi=center_inside,
        questions=questions,
        label_convention='zero-based center bins; half-open ROI; truncated visibility masked',
    )


def read_image(row):
    path = Path(row['image'][0])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
        raise ValueError('Prepared image changed')
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert('RGB')
    if list(image.size) != row['image_size']:
        raise ValueError('Prepared image dimensions changed')
    return image


def cell_for(box, roi, cells):
    x, y = [(box[i] + box[i + 2]) / 2 for i in (0, 1)]
    col = min(cells - 1, max(0, int((x - roi[0]) / (roi[2] - roi[0]) * cells)))
    row = min(cells - 1, max(0, int((y - roi[1]) / (roi[3] - roi[1]) * cells)))
    dx = (roi[2] - roi[0]) / cells
    dy = (roi[3] - roi[1]) / cells
    return [roi[0] + col * dx, roi[1] + row * dy, roi[0] + (col + 1) * dx, roi[1] + (row + 1) * dy]


def make_state(row, roi, depth, source, **extra):
    roi = [int(v) for v in roi]
    width, height = row['image_size']
    if not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
        raise ValueError('State ROI outside source screenshot')
    cells = 4 if depth == 0 else 8
    labels = labels_for_roi(row['bbox_xyxy'], roi, cells, local=depth > 0)
    return dict(
        target_id=row['id'],
        roi=roi,
        depth=depth,
        source=source,
        cells=cells,
        labels=labels,
        **extra,
    )


def synthetic_state(row, position, seed):
    from pixelone.search import expanded

    rng = random.Random(int(stable_hash([seed, position, 'roi'])[:16], 16))
    width, height = row['image_size']
    full = [0, 0, width, height]
    kind = position % 5
    if kind == 0:
        return make_state(row, full, 0, 'synthetic', sampling_kind='global')
    first = expanded(cell_for(row['bbox_xyxy'], full, 4), row['image_size'])
    depth = 1 if kind in (1, 3) else 2
    roi = first if depth == 1 else expanded(cell_for(row['bbox_xyxy'], first, 8), row['image_size'])
    rw, rh = roi[2] - roi[0], roi[3] - roi[1]
    if kind in (3, 4):
        for _ in range(32):
            x = rng.randint(0, width - rw)
            y = rng.randint(0, height - rh)
            trial = make_state(
                row, [x, y, x + rw, y + rh], depth, 'synthetic', sampling_kind='negative'
            )
            if trial['labels']['visibility_status'] == 'absent':
                return trial
        return make_state(
            row, roi, depth, 'synthetic', sampling_kind='negative_unavailable_positive_fallback'
        )
    for _ in range(16):
        x = max(0, min(width - rw, roi[0] + round(rng.uniform(-0.12, 0.12) * rw)))
        y = max(0, min(height - rh, roi[1] + round(rng.uniform(-0.12, 0.12) * rh)))
        trial = make_state(
            row, [x, y, x + rw, y + rh], depth, 'synthetic', sampling_kind='positive_jitter'
        )
        if trial['labels']['center_inside_roi']:
            return trial
    return make_state(row, roi, depth, 'synthetic', sampling_kind='positive_unjittered')


class StateSampler:
    """Select deterministic box-derived or cached visited states for teacher training."""

    def __init__(self, rows, recipe, rollout=None):
        self.rows = rows
        self.by_id = {r['id']: r for r in rows}
        self.recipe = recipe
        self.synthetic = SampleStream(len(rows), recipe['seed'])
        self.rollout = rollout or {}
        self.rollout_ids = sorted(self.rollout)
        self.real = (
            SampleStream(len(self.rollout_ids), recipe['seed']) if self.rollout_ids else None
        )

    def at(self, position):
        r = self.recipe
        use_real = (
            self.real is not None
            and int(stable_hash([r['seed'], position, 'mix'])[:16], 16) / 16**16
            < r['rollout_fraction']
        )
        if use_real:
            ident = self.rollout_ids[self.real.index(position)]
            states = self.rollout[ident]
            index = int(stable_hash([r['seed'], position, 'branch'])[:16], 16) % len(states)
            return self.by_id[ident], states[index]
        row = self.rows[self.synthetic.index(position)]
        return row, synthetic_state(row, position, r['seed'])


def prepare_state(model, processor, image, instruction, state, recipe):
    """Encode a teacher state and align its visibility, row, and column labels."""
    import torch
    from pixelone import views as geometry
    from pixelone.search import contextual_views, grid_questions
    from videojev_hf.processing_videojev import prepare_query_groups

    local = state['depth'] > 0
    size = recipe['crop_view_size'] if local else recipe['full_view_size']
    views = (
        contextual_views(geometry, image, state['roi'], state['cells'], size)
        if local
        else geometry.contextual_views(image, view_size=size)
    )
    # Encoding is frozen. Explicitly leave inference_mode and clone for autograd consumers.
    features, record = geometry.encode_views(model.native, processor, views, view_size=size)
    with torch.inference_mode(False):
        features = features.detach().clone()
    questions = grid_questions(geometry, instruction, state['cells'], local=local)
    groups = list(prepare_query_groups(processor, record, questions, record['context']))
    if len(groups) != 1:
        raise ValueError('All state questions must share exactly one native forward')
    group, actual = groups[0]
    target_questions = state['labels']['questions']
    if [q['id'] for q in actual] != [q['id'] for q in target_questions]:
        raise ValueError('Labels do not match native question order')
    labels = torch.tensor(
        [q['label_index'] if q['enabled'] else -100 for q in target_questions], dtype=torch.long
    )
    if not (labels != -100).any():
        raise ValueError('Unsuitable training state; no supervised question')
    frames = [hashlib.sha256(str(v.size).encode() + v.tobytes()).hexdigest() for v in views]
    return (
        group,
        features,
        labels,
        dict(
            frame_sha256=frames,
            visual_tokens=record['tokens'],
            input_tokens=group['input_ids'].shape[1],
        ),
    )
