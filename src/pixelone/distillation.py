"""Project teacher traces and match distributions only on observed support."""

import math

import torch
from torch.nn import functional as F

from pixelone.geometry import inside


def regions(prediction):
    """Collect every scored leaf from the teacher's final search round."""
    from pixelone.search import cell_candidates

    final = prediction['stages'][-1]
    leaves = []
    for trace in final['traces']:
        leaves.extend(
            cell_candidates(
                trace['crop_bounds'],
                trace['predictions'],
                final['cells_per_axis'],
                trace['parent_score'],
            )
        )
    if not leaves:
        raise ValueError('Teacher has no observed leaf cells')
    peak = max(x['score'] for x in leaves)
    return [(r['bounds'], math.exp(r['score'] - peak)) for r in leaves]


def project_regions(bounds_and_weights, point, box, layout):
    """Accumulate pixel-square overlap mass, gated by the teacher's final click."""
    if not inside(point, box):
        return dict(
            enabled=False,
            parents=torch.empty(0, dtype=torch.long),
            mass=torch.empty(0, layout.fine_count),
        )
    unit = layout.stride / layout.subdivisions
    indices = []
    weights = []
    for bounds, weight in bounds_and_weights:
        l, t, r, b = bounds
        if not all(math.isfinite(v) for v in (*bounds, weight)) or r <= l or b <= t or weight < 0:
            raise ValueError('Invalid teacher leaf')
        if weight == 0:
            continue
        x = torch.arange(
            max(0, math.floor(l / unit)),
            min(layout.cols * layout.subdivisions, math.ceil(r / unit)),
        )
        y = torch.arange(
            max(0, math.floor(t / unit)),
            min(layout.rows * layout.subdivisions, math.ceil(b / unit)),
        )
        x = x[(x + 0.5) * unit < layout.width]
        y = y[(y + 0.5) * unit < layout.height]
        if not len(x) or not len(y):
            continue
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        area = (
            torch.minimum((xx + 1) * unit, torch.tensor(float(r)))
            - torch.maximum(xx * unit, torch.tensor(float(l)))
        ).clamp_min(0) * (
            torch.minimum((yy + 1) * unit, torch.tensor(float(b)))
            - torch.maximum(yy * unit, torch.tensor(float(t)))
        ).clamp_min(
            0
        )
        parent = (yy // layout.subdivisions) * layout.cols + xx // layout.subdivisions
        fine = (yy % layout.subdivisions) * layout.subdivisions + xx % layout.subdivisions
        keep = area > 0
        indices.append((parent * layout.fine_count + fine)[keep])
        weights.append((area * weight / ((r - l) * (b - t)))[keep])
    if not indices:
        raise ValueError('Teacher support is not represented at this pixel resolution')
    flat = torch.cat(indices)
    values = torch.cat(weights).double()
    parents, inverse = torch.unique(flat // layout.fine_count, sorted=True, return_inverse=True)
    mass = torch.zeros(len(parents) * layout.fine_count, dtype=torch.float64)
    mass.scatter_add_(0, inverse * layout.fine_count + flat % layout.fine_count, values)
    mass = mass.reshape(len(parents), layout.fine_count)
    mass = mass / mass.sum()
    if not torch.isfinite(mass).all():
        raise ValueError('Nonfinite projected teacher mass')
    return dict(enabled=True, parents=parents, mass=mass.float(), point=list(point))


def project(prediction, box, layout):
    """Turn a cached search trace into normalized, sparse token/pixel supervision."""
    return project_regions(regions(prediction), prediction['predicted_xy'], box, layout)


def selected_parents(target, soft):
    """Include the GT-center token and, when enabled, teacher-observed tokens."""
    return [target['parent']] + (
        [int(v) for v in soft['parents'] if int(v) != target['parent']]
        if soft and soft['enabled']
        else []
    )


def kd_loss(output, soft, parent_order, temperature=2.0):
    """Reference two-level KL with observed-support masks and original token-mass weights."""
    coarse, fine = output['coarse'].float(), output['fine'].float()
    zero = coarse.sum() * 0 + fine.sum() * 0
    if not soft or not soft['enabled']:
        return zero, dict(kd_enabled=0, observed_parents=0, observed_pixels=0)
    parents = soft['parents'].to(coarse.device)
    mass = soft['mass'].to(coarse.device)
    coarse_mass = mass.sum(-1)
    soft_coarse = coarse_mass.pow(1 / temperature)
    soft_coarse /= soft_coarse.sum()
    rank = (
        F.kl_div(F.log_softmax(coarse[parents] / temperature, -1), soft_coarse, reduction='sum')
        * temperature**2
    )
    conditional = zero
    lookup = {int(p): i for i, p in enumerate(parent_order)}
    for i, parent in enumerate(parents.tolist()):
        mask = mass[i] > 0
        truth = mass[i, mask].pow(1 / temperature)
        truth /= truth.sum()
        values = fine[lookup[parent], mask]
        conditional = (
            conditional
            + coarse_mass[i]
            * F.kl_div(F.log_softmax(values / temperature, -1), truth, reduction='sum')
            * temperature**2
        )
    return rank + conditional, dict(
        kd_enabled=1, observed_parents=len(parents), observed_pixels=int((mass > 0).sum())
    )


def pixel_kd_loss(output, soft, parent_order, temperature=2.0):
    """Vectorized equivalent of kd_loss with the same masks, weights, and temperature."""
    coarse, fine = output['coarse'].float(), output['fine'].float()
    zero = coarse.sum() * 0 + fine.sum() * 0
    if not soft or not soft['enabled']:
        return zero, dict(kd_enabled=0, observed_parents=0, observed_pixels=0)
    parent_ids = soft['parents'].tolist()
    lookup = {parent: index for index, parent in enumerate(parent_order)}
    selected = torch.tensor([lookup[parent] for parent in parent_ids], device=fine.device)
    parents = soft['parents'].to(coarse.device)
    mass = soft['mass'].to(coarse.device)
    coarse_mass = mass.sum(-1)
    soft_coarse = coarse_mass.pow(1 / temperature)
    soft_coarse = soft_coarse / soft_coarse.sum()
    rank = (
        F.kl_div(F.log_softmax(coarse[parents] / temperature, -1), soft_coarse, reduction='sum')
        * temperature**2
    )
    mask = mass > 0
    truth = mass.pow(1 / temperature)
    truth = truth / truth.sum(-1, keepdim=True)
    log_student = F.log_softmax((fine[selected] / temperature).masked_fill(~mask, -torch.inf), -1)
    terms = F.kl_div(log_student.masked_fill(~mask, 0.0), truth, reduction='none')
    conditional = (terms.sum(-1) * coarse_mass).sum() * temperature**2
    return rank + conditional + zero, dict(
        kd_enabled=1,
        observed_parents=len(parent_ids),
        observed_pixels=int((soft['mass'] > 0).sum()),
    )
