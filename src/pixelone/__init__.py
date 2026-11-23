"""Public PixelOne package with a lazily imported inference API."""

__version__ = '0.1.0'


def __getattr__(name):
    if name == 'PixelOnePredictor':
        from .inference import PixelOnePredictor

        return PixelOnePredictor
    raise AttributeError(name)
