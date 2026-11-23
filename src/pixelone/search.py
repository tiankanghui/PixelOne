"""Three-round, two-candidate T3 search over row and column options."""

import copy
import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class T3Strategy:
    cells: tuple = (4, 8, 8)
    groups: int = 5
    questions: int = 14
    calls: int = 3


T3_STRATEGY = T3Strategy()


def expanded(bounds, image_size):
    left, top, right, bottom = bounds
    horizontal, vertical = (right - left) * 0.12, (bottom - top) * 0.12
    return [
        max(0, math.floor(left - horizontal)),
        max(0, math.floor(top - vertical)),
        min(image_size[0], math.ceil(right + horizontal)),
        min(image_size[1], math.ceil(bottom + vertical)),
    ]


def center(bounds):
    return [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]


def grid_questions(geometry, instruction, cells, local=False):
    questions = copy.deepcopy(geometry.grid_questions(instruction, local=local))
    if cells == 4:
        return questions
    if cells != 8:
        raise ValueError('Only four/eight-way local choices are supported')
    letters = 'ABCDEFGH'
    questions[-2]['options'] = [
        f'{letter}. Column {letter}, column {index + 1} of {cells} from the left.'
        for index, letter in enumerate(letters)
    ]
    questions[-1]['options'] = [
        f'{letter}. Row {index + 1}, row {index + 1} of {cells} from the top.'
        for index, letter in enumerate(letters)
    ]
    return questions


def crop_canvas(image, bounds, cells, view_size, font_path):
    if cells != 8 or type(view_size) is not int or view_size < 128:
        raise ValueError('Eight-way grids require a fixed-size crop canvas')
    source = image.crop(tuple(round(value) for value in bounds))
    scale = view_size / 768
    factor = min(720 / source.width, 692 / source.height)
    size = (max(1, round(source.width * factor)), max(1, round(source.height * factor)))
    content = source.resize(size, Image.Resampling.LANCZOS)
    left, top = round((36 + (720 - content.width) // 2) * scale), round(58 * scale)
    if view_size != 768:
        content = source.resize(
            (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new('RGB', (view_size, view_size), 'white')
    canvas.paste(content, (left, top))
    font = ImageFont.truetype(str(font_path), round(21 * scale))
    ImageDraw.Draw(canvas).text(
        (round(20 * scale), round(7 * scale)),
        'ENLARGED CROP - answer for this 8x8 grid',
        fill='#142c48',
        font=font,
    )
    overlay = Image.new('RGBA', canvas.size)
    draw = ImageDraw.Draw(overlay)
    for divider in range(1, cells):
        horizontal, vertical = (
            left + content.width * divider / cells,
            top + content.height * divider / cells,
        )
        draw.line(
            (horizontal, top, horizontal, top + content.height),
            fill=(126, 52, 191, 155),
            width=max(1, round(2 * scale)),
        )
        draw.line(
            (left, vertical, left + content.width, vertical),
            fill=(0, 145, 168, 155),
            width=max(1, round(2 * scale)),
        )
    canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(canvas)
    for index, letter in enumerate('ABCDEFGH'):
        horizontal = left + content.width * (index + 0.5) / cells
        vertical = top + content.height * (index + 0.5) / cells
        draw.text((horizontal - 8 * scale, top - 27 * scale), letter, fill='#7135b0', font=font)
        draw.text(
            (left - 24 * scale, vertical - 11 * scale), str(index + 1), fill='#007f98', font=font
        )
    draw.rectangle(
        (left, top, left + content.width, top + content.height),
        outline='#738090',
        width=max(1, round(scale)),
    )
    return canvas


def contextual_views(geometry, image, bounds, cells, view_size):
    if cells == 4:
        return geometry.contextual_views(image, bounds, view_size=view_size)
    renderer = getattr(geometry, 'view_canvas', None) or geometry.render_view
    font_path = getattr(geometry, 'FONT_PATH', None) or geometry.FONT
    overview = image.copy()
    ImageDraw.Draw(overview).rectangle(bounds, outline='#f38d16', width=max(3, image.width // 180))
    full = renderer(overview, title='FULL SCREEN - orange box is the crop', view_size=view_size)
    local = crop_canvas(image, bounds, cells, view_size, font_path)
    return [full, full, local, local]


def cell_candidates(bounds, predictions, cells, parent_score=0.0):
    """Form grid cells and rank them by accumulated row, column, and visibility scores."""
    expected = [2, cells, cells] if len(predictions) == 3 else [cells, cells]
    if len(predictions) not in (2, 3):
        raise ValueError('Expected two coarse or three crop questions')
    for prediction, count in zip(predictions, expected, strict=True):
        values = prediction['probabilities']
        if (
            len(values) != count
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-4)
        ):
            raise ValueError('Invalid choice probabilities or grid size')
    horizontal, vertical = predictions[-2:]
    presence = predictions[0]['probabilities'][0] if len(predictions) == 3 else 1.0
    left, top, right, bottom = bounds
    result = []
    for row_index in range(cells):
        for column in range(cells):
            probability = horizontal['probabilities'][column] * vertical['probabilities'][row_index]
            cell = [
                left + (right - left) * column / cells,
                top + (bottom - top) * row_index / cells,
                left + (right - left) * (column + 1) / cells,
                top + (bottom - top) * (row_index + 1) / cells,
            ]
            result.append(
                dict(
                    bounds=cell,
                    column=column,
                    row=row_index,
                    presence_probability=presence,
                    score=parent_score
                    + math.log(max(probability, 1e-12))
                    + math.log(max(presence, 1e-12)),
                )
            )
    return sorted(result, key=lambda candidate: candidate['score'], reverse=True)


def refine_batch(backend, images, instructions, full_views, crop_size):
    """Run T3 with two retained candidates and return clicks plus complete search traces."""
    strategy = T3_STRATEGY
    count = len(images)
    if not count or any(len(values) != count for values in (instructions, full_views)):
        raise ValueError('Provide matching nonempty images, instructions, views')
    geometry = backend.geometry
    predictions = backend.score_many(
        [
            (view, grid_questions(geometry, instruction, strategy.cells[0]))
            for view, instruction in zip(full_views, instructions, strict=True)
        ]
    )
    beams = [
        cell_candidates([0, 0, *image.size], choices, strategy.cells[0])[:2]
        for image, choices in zip(images, predictions, strict=True)
    ]
    stages = [
        [
            dict(
                depth=0,
                cells_per_axis=strategy.cells[0],
                predicted_xy=center(beam[0]['bounds']),
                candidates=beam,
                predictions=choices,
            )
        ]
        for beam, choices in zip(beams, predictions, strict=True)
    ]
    for depth, cells in enumerate(strategy.cells[1:], start=1):
        requests, records = [], []
        for rank in range(2):
            for index, (image, instruction, beam) in enumerate(
                zip(images, instructions, beams, strict=True)
            ):
                parent = beam[rank]
                bounds = expanded(parent['bounds'], image.size)
                encoded = backend.encode(
                    contextual_views(geometry, image, bounds, cells, crop_size), crop_size
                )
                requests.append((encoded, grid_questions(geometry, instruction, cells, local=True)))
                records.append(
                    (
                        index,
                        dict(
                            parent_rank=rank,
                            parent_bounds=parent['bounds'],
                            parent_score=parent['score'],
                            crop_bounds=bounds,
                            encoding=backend.describe(encoded),
                        ),
                    )
                )
        choices = backend.score_many(requests)
        children, traces = [[] for _image in images], [[] for _image in images]
        for (index, trace), predictions in zip(records, choices, strict=True):
            trace['predictions'] = predictions
            children[index].extend(
                cell_candidates(trace['crop_bounds'], predictions, cells, trace['parent_score'])
            )
            traces[index].append(trace)
        beams = [
            sorted(items, key=lambda candidate: candidate['score'], reverse=True)[:2]
            for items in children
        ]
        for target_stages, beam, trace in zip(stages, beams, traces, strict=True):
            target_stages.append(
                dict(
                    depth=depth,
                    cells_per_axis=cells,
                    predicted_xy=center(beam[0]['bounds']),
                    candidates=beam,
                    traces=trace,
                )
            )
    results = []
    for target_stages, beam in zip(stages, beams, strict=True):
        point = target_stages[-1]['predicted_xy']
        results.append(
            dict(
                stages=target_stages,
                unsnapped_xy=point,
                final_presence_probability=beam[0]['presence_probability'],
                logical_language_groups=strategy.groups,
                logical_questions=strategy.questions,
                predicted_xy=list(point),
                snap_candidate_index=None,
                snap_distance_px=0.0,
                snap_changed=False,
                snap_max_fraction=0.0,
            )
        )
    return results
