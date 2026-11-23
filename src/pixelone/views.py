"""Render teacher views and encode them with the native video interface."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers.video_utils import VideoMetadata
from videojev_hf.videojev_temporal import pair_timeline

FONT_PATH = str(Path(__file__).resolve().parent / 'assets' / 'DejaVuSans.ttf')
VIEW_SIZE = 768
NATIVE_ALIGNMENT = 32
QUERY_CONTEXT = 8192
NATIVE_QUERY_CONTEXT = 32768


def native_canvas_size(image_size):
    if len(image_size) != 2 or any(not isinstance(size, int) or size < 1 for size in image_size):
        raise ValueError('Provide positive integer image width and height')
    width, height = image_size
    return (
        math.ceil((width + 48) / NATIVE_ALIGNMENT) * NATIVE_ALIGNMENT,
        math.ceil((height + 76) / NATIVE_ALIGNMENT) * NATIVE_ALIGNMENT,
    )


def view_canvas(image, bounds=None, marked=False, title='FULL SCREEN', view_size=VIEW_SIZE):
    if view_size != 'native' and (not isinstance(view_size, int) or view_size < 128):
        raise ValueError('View size must be an integer >= 128')
    if view_size == 'native':
        if bounds is not None:
            raise ValueError('Native resolution is only supported for the initial full screenshot')
        content = image.copy()
        scale, left, top = 1, 36, 58
        canvas_size = native_canvas_size(image.size)
    else:
        scale = view_size / VIEW_SIZE
        source = image if bounds is None else image.crop(tuple(round(value) for value in bounds))
        content = source.copy()
        content.thumbnail((VIEW_SIZE - 48, VIEW_SIZE - 76), Image.Resampling.LANCZOS)
        if bounds is not None:
            crop_scale = min((VIEW_SIZE - 48) / content.width, (VIEW_SIZE - 76) / content.height)
            if crop_scale > 1:
                content = content.resize(
                    (round(content.width * crop_scale), round(content.height * crop_scale)),
                    Image.Resampling.LANCZOS,
                )
        left, top = round((36 + (VIEW_SIZE - 48 - content.width) // 2) * scale), round(58 * scale)
        if view_size != VIEW_SIZE:
            content = source.resize(
                (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
                Image.Resampling.LANCZOS,
            )
        canvas_size = (view_size, view_size)
    canvas = Image.new('RGB', canvas_size, 'white')
    canvas.paste(content, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, round(21 * scale))
    draw.text((round(20 * scale), round(7 * scale)), title, fill='#142c48', font=font)
    if marked:
        overlay = Image.new('RGBA', canvas.size)
        grid_draw = ImageDraw.Draw(overlay)
        for divider in range(1, 4):
            horizontal = left + content.width * divider / 4
            vertical = top + content.height * divider / 4
            grid_draw.line(
                (horizontal, top, horizontal, top + content.height),
                fill=(126, 52, 191, 155),
                width=max(1, round(2 * scale)),
            )
            grid_draw.line(
                (left, vertical, left + content.width, vertical),
                fill=(0, 145, 168, 155),
                width=max(1, round(2 * scale)),
            )
        canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(canvas)
        for index in range(4):
            horizontal = left + content.width * (index + 0.5) / 4
            vertical = top + content.height * (index + 0.5) / 4
            draw.text(
                (horizontal - 8 * scale, top - 27 * scale), 'ABCD'[index], fill='#7135b0', font=font
            )
            draw.text(
                (left - 24 * scale, vertical - 11 * scale),
                str(index + 1),
                fill='#007f98',
                font=font,
            )
        draw.rectangle(
            (left, top, left + content.width, top + content.height),
            outline='#738090',
            width=max(1, round(scale)),
        )
    return canvas


def contextual_views(image, bounds=None, view_size=VIEW_SIZE):
    if bounds is None:
        view = view_canvas(
            image, marked=True, title='FULL SCREEN - labeled grid', view_size=view_size
        )
        return [view, view]
    if view_size == 'native':
        raise ValueError('Native resolution is only supported for the initial full screenshot')
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    draw.rectangle(bounds, outline='#f38d16', width=max(3, image.width // 180))
    full = view_canvas(overview, title='FULL SCREEN - orange box is the crop', view_size=view_size)
    local = view_canvas(
        image,
        bounds,
        marked=True,
        title='ENLARGED CROP - answer for this grid',
        view_size=view_size,
    )
    return [full, full, local, local]


def grid_questions(instruction, local=False):
    description = (
        'The first view is the full screenshot for context. The last view is an enlarged crop. '
        'Answer about the labeled grid in the LAST enlarged crop, not the full screenshot. '
        if local
        else 'The frames show one GUI screenshot with a labeled grid. '
    )
    prefix = description + f'Locate the UI element to click for this instruction: "{instruction}". '
    questions = [
        dict(
            id='horizontal',
            task='choice',
            query=prefix + 'Which labeled COLUMN contains the center of the target?',
            options=[
                'A. Column A, the leftmost column.',
                'B. Column B, the middle-left column.',
                'C. Column C, the middle-right column.',
                'D. Column D, the rightmost column.',
            ],
        ),
        dict(
            id='vertical',
            task='choice',
            query=prefix + 'Which labeled ROW contains the center of the target?',
            options=[
                'A. Row 1, the topmost row.',
                'B. Row 2, the upper-middle row.',
                'C. Row 3, the lower-middle row.',
                'D. Row 4, the bottommost row.',
            ],
        ),
    ]
    if local:
        questions.insert(
            0,
            dict(
                id='presence',
                task='choice',
                query=prefix + 'Is the target UI element visible inside the LAST enlarged crop?',
                options=[
                    'A. Yes, the target is visible in the enlarged crop.',
                    'B. No, the target is not visible in the enlarged crop.',
                ],
            ),
        )
    return questions


def encode_views(model, processor, views, view_size=VIEW_SIZE):
    if view_size != 'native' and (not isinstance(view_size, int) or view_size < 1):
        raise ValueError('View size must be a positive integer')
    count = len(views)
    options = (
        dict(size=dict(shortest_edge=4096, longest_edge=count * view_size**2))
        if view_size != 'native'
        else {}
    )
    if view_size == 'native':
        if not views or len({view.size for view in views}) != 1:
            raise ValueError('Native frames must have one matching nonempty canvas size')
        width, height = views[0].size
        video_processor = processor.video_processor
        if (
            video_processor.patch_size * video_processor.merge_size != NATIVE_ALIGNMENT
            or width % NATIVE_ALIGNMENT
            or height % NATIVE_ALIGNMENT
        ):
            raise ValueError(
                'Native canvases must be padded to the checkpoint\'s 32-pixel alignment'
            )
        options = dict(
            do_resize=False, size=dict(shortest_edge=4096, longest_edge=count * width * height)
        )
    metadata = VideoMetadata(total_num_frames=count, fps=2.0, frames_indices=list(range(count)))
    native = processor.video_processor(
        videos=[np.stack([np.asarray(view) for view in views])],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors='pt',
        **options,
    )
    grid = native['video_grid_thw'][0].tolist()
    if view_size == 'native':
        expected = [
            math.ceil(count / video_processor.temporal_patch_size),
            height // video_processor.patch_size,
            width // video_processor.patch_size,
        ]
        if grid != expected:
            raise ValueError(
                f'Native full resolution was changed by the processor: {grid} != {expected}'
            )
    times, bounds = pair_timeline(np.arange(count) / 2.0, count / 2.0)
    record = dict(
        frames=count,
        fps=2.0,
        grid_thw=grid,
        tokens=int(np.prod(grid)) // 4,
        view_size=view_size,
        canvas_size=list(views[0].size),
        processor_resize=view_size != 'native',
        context=NATIVE_QUERY_CONTEXT if view_size == 'native' else QUERY_CONTEXT,
        score_times=times.tolist(),
        score_bounds=bounds.tolist(),
        sampled_duration=count / 2.0,
    )
    if record['tokens'] >= record['context']:
        raise ValueError(
            'Visual tokens leave no room for questions; no automatic downscaling is allowed'
        )
    features = model.encode_video(native['pixel_values_videos'], torch.tensor([grid]))
    return features, record
