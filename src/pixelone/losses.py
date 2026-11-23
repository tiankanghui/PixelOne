"""Top20 box supervision combined with teacher-gated spatial distillation."""

import torch
from torch.nn import functional as F

from pixelone.model import route
from pixelone.distillation import kd_loss, pixel_kd_loss


def support(layout, box, device=None):
    # Reuse strict GT validation; representability is checked against the full set.
    """Return original-image and target-box masks on the pixel-center grid."""
    layout.center_target(box)
    xy = layout.positions(device)
    valid = layout.valid(device)
    positive = (
        valid
        & (xy[..., 0] >= box[0])
        & (xy[..., 0] <= box[2])
        & (xy[..., 1] >= box[1])
        & (xy[..., 1] <= box[3])
    )
    if not positive.any():
        raise ValueError('GT box contains no representable pixel center')
    return valid, positive


def set_nll(logits, valid, positive):
    """Negative log probability of a nonempty set of valid target choices."""
    if logits.shape != valid.shape or valid.shape != positive.shape:
        raise ValueError('Click support shapes differ')
    if (positive & ~valid).any() or not positive.any(-1).all():
        raise ValueError('Positive click support must be nonempty and valid')
    logits = logits.float()
    return logits.masked_fill(~valid, -torch.inf).logsumexp(-1) - logits.masked_fill(
        ~positive, -torch.inf
    ).logsumexp(-1)


def top20_winner_loss(logits, valid, positive):
    """Keep the original negative pool and token weighting; expand positive winners.

    Each selected positive competes separately against every negative. Selection
    is detached, but gradients flow through every gathered positive score. The
    final decoder remains the original token-first, one-pixel argmax.
    """
    if (
        logits.ndim != 2
        or logits.shape != valid.shape
        or valid.shape != positive.shape
        or valid.dtype != torch.bool
        or positive.dtype != torch.bool
    ):
        raise ValueError('Top20 support shapes/types differ')
    if not positive.any(-1).all() or (positive & ~valid).any():
        raise ValueError('Every fine row needs nonempty valid GT support')
    z = logits.float()
    negative = valid & ~positive
    active = negative.any(-1)
    zero = z.sum() * 0
    if not active.any():
        return zero

    selected, pos, neg = z[active], positive[active], negative[active]
    # Integer arithmetic gives exact ceil(20% * n_positive), including n < 5.
    counts = (pos.sum(-1) + 4) // 5
    max_count = int(counts.max())
    order = (
        selected.detach()
        .masked_fill(~pos, -torch.inf)
        .argsort(dim=-1, descending=True, stable=True)[:, :max_count]
    )
    keep = torch.arange(max_count, device=z.device)[None] < counts[:, None]
    best = selected.gather(-1, order).masked_fill(~keep, 0.0)
    wrong = selected.masked_fill(~neg, -torch.inf).logsumexp(-1)
    per_positive = F.softplus(wrong[:, None] - best).masked_fill(~keep, 0.0)
    per_token = per_positive.sum(-1) / counts
    # Fully GT-covered tokens retain their original zero-loss contribution.
    return per_token.sum() / len(z) + zero


def gt_objective(output, valid, positive, target, layout, topk):
    """Keep box-supervised routing independent of additional KD readouts."""
    parents = route(output['coarse'], [target['parent']], positive.any(-1), topk)
    lookup = {int(parent): index for index, parent in enumerate(output['parents'].tolist())}
    if any(int(parent) not in lookup for parent in parents):
        raise ValueError('GT routing is absent from the shared forward')
    indices = torch.tensor(
        [lookup[int(parent)] for parent in parents], device=output['fine'].device
    )
    view = dict(coarse=output['coarse'], fine=output['fine'][indices], parents=parents)
    coarse = set_nll(view['coarse'], valid.any(-1), positive.any(-1))
    truth = positive[parents]
    keep = truth.any(-1)
    if not keep.any():
        raise ValueError('No supervised positive region was routed')
    fine = top20_winner_loss(view['fine'][keep], valid[parents][keep], truth[keep])
    total = coarse + fine
    terms = dict(
        coarse=float(coarse.detach()), fine=float(fine.detach()), supervised_regions=int(keep.sum())
    )
    return total, terms, parents.tolist()


def unified_loss(output, valid, positive, target, layout, method, soft, recipe):
    """Combine the fixed Top20 box objective with optional spatial KD."""
    if method not in ('gt_top20', 'pixel_kd_top20'):
        raise ValueError('Public training supports Top20 with spatial KD or GT only')
    gt, terms, gt_parents = gt_objective(
        output, valid, positive, target, layout, recipe['routing_topk']
    )
    parents = output['parents'].tolist()
    if method == 'gt_top20':
        if soft is not None:
            raise ValueError('GT-only must not use teacher prediction/support')
        kd = output['coarse'].float().sum() * 0 + output['fine'].float().sum() * 0
        kd_terms = dict(kd_enabled=0, observed_parents=0, observed_pixels=0)
        details = dict(method=method, coarse=0.0, fine=0.0)
    else:
        implementation = (
            pixel_kd_loss if recipe.get('speed', {}).get('vectorized_kd', False) else kd_loss
        )
        kd, kd_terms = implementation(output, soft, parents, recipe['temperature'])
        details = dict(method='pixel_kd')
    loss = gt + recipe['kd_weight'] * kd
    return loss, gt, terms, kd, kd_terms, details, gt_parents
