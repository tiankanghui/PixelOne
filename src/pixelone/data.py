"""Validate grounding targets and bind teacher traces to their source data."""

import json
from pathlib import Path
import sqlite3

from PIL import Image

from pixelone.geometry import Layout
from pixelone.losses import support
from pixelone.io import file_sha256


def normalize_row(row, image_root, verify_image=True):
    required = {'id', 'image', 'instruction', 'bbox_xyxy'}
    if not required <= row.keys():
        raise ValueError('Missing fields: ' + ', '.join(sorted(required - row.keys())))
    if not isinstance(row['id'], str) or not row['id']:
        raise ValueError('Target IDs must be nonempty strings')
    if (
        not isinstance(row['instruction'], str)
        or not row['instruction'].strip()
        or '<|' in row['instruction']
    ):
        raise ValueError('Expected a nonempty instruction without model control tokens')
    image = row['image']
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError('Each target must reference one screenshot')
        image = image[0]
    path = Path(image_root) / image
    if verify_image or 'image_size' not in row or 'image_sha256' not in row:
        with Image.open(path) as source:
            dimensions = list(source.size)
        digest = file_sha256(path)
        if (
            row.get('image_size', dimensions) != dimensions
            or row.get('image_sha256', digest) != digest
        ):
            raise ValueError('Screenshot identity changed: ' + row['id'])
    else:
        dimensions, digest = row['image_size'], row['image_sha256']
    layout = Layout(*dimensions)
    support(layout, row['bbox_xyxy'])
    return {**row, 'image': [str(path)], 'image_size': dimensions, 'image_sha256': digest}


class GroundingDataset:
    """Read individual grounding rows by byte offset and verify screenshot identity."""

    def __init__(self, path, image_root=None):
        self.path = Path(path)
        self.image_root = Path(image_root) if image_root else self.path.parent
        self.offsets, identifiers = [], set()
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                identifier = row['id']
                if identifier in identifiers:
                    raise ValueError('Duplicate target ID: ' + identifier)
                identifiers.add(identifier)
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError('Empty grounding dataset')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return normalize_row(row, self.image_root)


def prepare_dataset(source, destination, image_root=None):
    """Validate original-pixel annotations and write a new, non-overwriting JSONL."""
    source, destination = Path(source), Path(destination)
    root = Path(image_root) if image_root else source.parent
    if destination.exists():
        raise ValueError('Choose a new output path; datasets are not overwritten')
    destination.parent.mkdir(parents=True, exist_ok=True)
    identifiers = set()
    with source.open() as inputs, destination.open('x') as outputs:
        for line in inputs:
            original = json.loads(line)
            row = normalize_row(original, root)
            if row['id'] in identifiers:
                raise ValueError('Duplicate target ID: ' + row['id'])
            identifiers.add(row['id'])
            row['image'] = original['image']
            outputs.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return {'targets': len(identifiers), 'sha256': file_sha256(destination)}


def index_teacher_cache(path, destination):
    path, destination = Path(path), Path(destination)
    if destination.exists():
        raise ValueError('Choose a new index path')
    connection = sqlite3.connect(destination)
    try:
        connection.execute(
            'CREATE TABLE records (id TEXT PRIMARY KEY, offset INTEGER, length INTEGER)'
        )
        connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
        with path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                connection.execute(
                    'INSERT INTO records VALUES (?, ?, ?)', (row['id'], offset, len(line))
                )
        connection.execute('INSERT INTO metadata VALUES (?, ?)', ('sha256', file_sha256(path)))
        connection.commit()
    finally:
        connection.close()


class IndexedTeacherCache:
    """Look up immutable teacher traces and verify their target and image identity."""

    def __init__(self, path, index):
        self.path = Path(path)
        self.connection = sqlite3.connect(Path(index).resolve().as_uri() + '?mode=ro', uri=True)
        expected = self.connection.execute(
            'SELECT value FROM metadata WHERE key="sha256"'
        ).fetchone()
        if expected is None or file_sha256(self.path) != expected[0]:
            raise ValueError('Teacher cache does not match its index')

    def get(self, target):
        offset = self.connection.execute(
            'SELECT offset, length FROM records WHERE id=?', (target['id'],)
        ).fetchone()
        if offset is None:
            raise ValueError('No teacher trace for target: ' + target['id'])
        with self.path.open('rb') as stream:
            stream.seek(offset[0])
            row = json.loads(stream.read(offset[1]))
        if (
            any(row[key] != target[key] for key in ['id', 'instruction', 'bbox_xyxy', 'image_size'])
            or row['source_image_sha256'] != target['image_sha256']
        ):
            raise ValueError('Teacher trace and training input identities differ')
        return row

    def close(self):
        self.connection.close()
