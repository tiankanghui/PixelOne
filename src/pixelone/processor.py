"""Adapt Qwen processor inputs to the native VideoJEV decision interface."""

import numpy as np
import torch


class LegacyProcessorAdapter:
    """Expose refactored prompt helpers using the native 5.6.2 formatting."""

    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def replace_video_token(self, video_inputs, *, video_idx):
        grid = video_inputs['video_grid_thw'][video_idx]
        metadata = video_inputs['video_metadata'][video_idx]
        timestamps = self._calculate_timestamps(
            list(metadata.frames_indices), metadata.fps, self.video_processor.temporal_patch_size
        )
        spatial = int(np.prod(grid[1:])) // self.video_processor.merge_size**2
        return ''.join(
            f'<{timestamp:.1f} seconds>{self.vision_start_token}'
            f'{self.video_token * spatial}{self.vision_end_token}'
            for timestamp in timestamps[: int(grid[0])]
        )

    def get_text_with_replacements(self, texts, images, videos, audio):
        if len(texts) != 1 or len(videos) != 1 or images or audio:
            raise ValueError('This compatibility adapter supports one video-only prompt')
        placeholder = self.vision_start_token + self.video_token + self.vision_end_token
        if placeholder not in texts[0]:
            placeholder = self.video_token
        if placeholder not in texts[0]:
            raise ValueError('No native video placeholder in the prompt')
        return [texts[0].replace(placeholder, videos[0], 1)], None


def legacy_unpadded_mask(_module, positional_args, keyword_args):
    """Let 5.6.2 build its native causal mask for a single unpadded sequence."""
    mask = keyword_args.get('attention_mask')
    if isinstance(mask, dict):
        if any(value is not None for value in mask.values()):
            raise ValueError('The legacy adapter only supports an unpadded attention mask')
        keyword_args['attention_mask'] = None
    return positional_args, keyword_args
