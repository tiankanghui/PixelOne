"""Deterministic shuffled sample streams shared across training methods."""


class SampleStream:
    """Deterministic epoch permutations indexed by absolute sample position."""

    def __init__(self, count, seed):
        if count < 1:
            raise ValueError('Empty training dataset')
        self.count = count
        self.seed = seed
        self.epoch = None
        self.order = None

    def index(self, global_position):
        import numpy as np

        epoch, position = divmod(global_position, self.count)
        if epoch != self.epoch:
            self.order = np.random.default_rng(self.seed + epoch).permutation(self.count)
            self.epoch = epoch
        return int(self.order[position])
