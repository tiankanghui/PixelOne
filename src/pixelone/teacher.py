"""Connect T3 search to a trained teacher and cache its spatial traces."""

import json
from pathlib import Path

import torch

from pixelone.runtime import native_runtime
from pixelone.states import make_state, read_image
from pixelone.search import refine_batch
from videojev_hf.processing_videojev import prepare_query_groups


class SearchTeacher:
    """Adapter from the trained decision model to the T3 search backend."""

    def __init__(self, model, processor, recipe):
        from pixelone import views

        self.model, self.processor, self.recipe = model, processor, recipe
        self.geometry = views

    def encode(self, views, view_size):
        features, record = self.geometry.encode_views(
            self.model.native, self.processor, views, view_size
        )
        with torch.inference_mode(False):
            features = features.detach().clone()
        return features, record

    def describe(self, encoded):
        return encoded[1]

    def score_many(self, requests):
        groups, features, questions = [], [], []
        for encoded, requested in requests:
            feature, record = encoded
            prepared = list(
                prepare_query_groups(self.processor, record, requested, record['context'])
            )
            if len(prepared) != 1:
                raise ValueError('A search state must fit in one model call')
            group, actual = prepared[0]
            groups.append(group)
            features.append(feature)
            questions.append(actual)
        outputs = self.model(groups, features)['items']
        return [
            [
                {
                    'id': question['id'],
                    'probabilities': output['probabilities'][index, : len(question['options'])]
                    .float()
                    .cpu()
                    .tolist(),
                }
                for index, question in enumerate(actual)
            ]
            for output, actual in zip(outputs, questions, strict=True)
        ]

    def predict(self, image, instruction):
        previous = self.model.training
        self.model.eval()
        try:
            with native_runtime(self.recipe['kernel_backend']), torch.no_grad():
                full = self.encode(
                    self.geometry.contextual_views(image, view_size=self.recipe['full_view_size']),
                    self.recipe['full_view_size'],
                )
                return refine_batch(
                    self, [image], [instruction], [full], self.recipe['crop_view_size']
                )[0]
        finally:
            self.model.train(previous)


def visited_states(row, prediction):
    result = [make_state(row, [0, 0, *row['image_size']], 0, 'visited')]
    for stage in prediction['stages'][1:]:
        for trace in stage['traces']:
            state = make_state(row, trace['crop_bounds'], stage['depth'], 'visited')
            if any(question['enabled'] for question in state['labels']['questions']):
                result.append(state)
    return result


def cache_teacher(teacher, rows, destination):
    """Write immutable search traces and a companion SQLite byte-offset index."""
    from pixelone.data import index_teacher_cache

    destination = Path(destination)
    if destination.exists():
        raise ValueError('Choose a new teacher-cache path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        for row in rows:
            prediction = teacher.predict(read_image(row), row['instruction'])
            record = {
                **prediction,
                **{key: row[key] for key in ['id', 'instruction', 'bbox_xyxy', 'image_size']},
                'source_image_sha256': row['image_sha256'],
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    index_teacher_cache(destination, destination.with_suffix('.sqlite'))
