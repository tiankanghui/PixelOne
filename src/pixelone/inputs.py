"""One static native screenshot, one query, spatial visual-token candidates."""

import hashlib

import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from pixelone.geometry import Layout


def render(image, subdivisions=32):
    """Pad on the right and bottom without resizing the original screenshot."""
    layout = Layout(*image.size, subdivisions=subdivisions)
    canvas = Image.new('RGB', layout.canvas_size, 'white')
    canvas.paste(image.convert('RGB'), (0, 0))
    return [canvas, canvas], layout


def query_group(processor, record, instruction):
    from videojev_hf.processing_videojev import DECIDE

    if not instruction.strip() or '<|' in instruction:
        raise ValueError('Expected an instruction without model control tokens')
    if record['grid_thw'][0] != 1:
        raise ValueError('Spatial pointer uses exactly one static frame pair')
    metadata = VideoMetadata(total_num_frames=2, fps=2.0, frames_indices=[0, 1])
    replacement = processor.replace_video_token(
        dict(video_grid_thw=np.asarray([[1, 2, 2]]), video_metadata=[metadata]), video_idx=0
    )
    content = [
        dict(type='video', video='provided_in_memory'),
        dict(
            type='text',
            text=(
                f'Locate the UI element for this instruction: "{instruction}". '
                'Select a valid click position inside that element in the screenshot.\n' + DECIDE
            ),
        ),
    ]
    text = processor.apply_chat_template(
        [dict(role='user', content=content)], tokenize=False, add_generation_prompt=False
    )
    texts, _ = processor.get_text_with_replacements([text], [], [replacement], [])
    encoded = processor(text=texts, return_tensors='pt')
    ids, types = encoded['input_ids'], encoded['mm_token_type_ids']
    slots = ids[0] == processor.video_token_id
    if int(slots.sum()) != 1:
        raise ValueError('Expected one temporary video slot')
    repeats = torch.ones(ids.shape[1], dtype=torch.long)
    repeats[slots] = record['tokens']
    ids = torch.repeat_interleave(ids, repeats, dim=1)
    types = torch.repeat_interleave(types, repeats, dim=1)
    spatial = (ids[0] == processor.video_token_id).nonzero().flatten()
    query = (ids[0] == processor.tokenizer.convert_tokens_to_ids(DECIDE)).nonzero().flatten()
    if query.numel() != 1 or spatial.numel() != record['tokens'] or spatial[-1] >= query[0]:
        raise ValueError('Spatial/query slots are not intact')
    if (ids.shape[1] + 63) // 64 * 64 > record['context']:
        raise ValueError('Native image and instruction exceed context; no silent resize')
    return dict(
        input_ids=ids,
        mm_token_type_ids=types,
        video_grid_thw=torch.tensor([record['grid_thw']]),
        pair_indices=spatial[None],
        query_indices=query,
        score_mask=torch.ones((1, len(spatial)), dtype=torch.bool),
        decision_tasks=torch.ones(1, dtype=torch.long),
    )


def prepare(model, processor, image, instruction, subdivisions=32):
    """Encode a screenshot and instruction for one direct student forward."""
    from pixelone.views import encode_views

    views, layout = render(image, subdivisions)
    features, record = encode_views(model.native, processor, views, 'native')
    if record['tokens'] != layout.count or record['grid_thw'] != [
        1,
        layout.rows * 2,
        layout.cols * 2,
    ]:
        raise ValueError('Native visual grid does not match original-pixel support')
    if features.is_inference() or not getattr(model.native, 'train_vision', False):
        with torch.inference_mode(False):
            features = features.detach().clone()
    group = query_group(processor, record, instruction)
    return (
        group,
        features,
        layout,
        dict(
            visual_tokens=record['tokens'],
            input_tokens=group['input_ids'].shape[1],
            frames=2,
            canvas_size=list(layout.canvas_size),
            original_size=list(image.size),
            padding_origin=[0, 0],
            frame_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in views],
        ),
    )
