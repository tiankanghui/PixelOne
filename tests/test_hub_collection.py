import json

import pytest

from pixelone.inference import resolve_model_path


def test_local_model_subfolder(tmp_path):
    model = tmp_path / 'PixelOne-Pro-4B'
    model.mkdir()
    assert resolve_model_path(tmp_path, subfolder='PixelOne-Pro-4B') == model
    assert resolve_model_path(model) == model
    with pytest.raises(FileNotFoundError):
        resolve_model_path(tmp_path, subfolder='missing')


@pytest.mark.parametrize('folder', ['../private', '/absolute', 'nested/../../private', '..\\private', '.'])
def test_unsafe_model_subfolders_are_rejected(tmp_path, folder):
    with pytest.raises(ValueError):
        resolve_model_path(tmp_path, subfolder=folder)


@pytest.mark.parametrize('folder', [None, 'PixelOne-Pro-4B', 'PixelOne-Pro-35B-A3B'])
def test_hub_download_requests_only_the_selected_models_files(tmp_path, monkeypatch, folder):
    import huggingface_hub
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'config.json': 'hash', 'weights-00001.safetensors': 'hash'}}))
    calls = []
    def download_manifest(repo_id, **kwargs):
        calls.append(('manifest', repo_id, kwargs))
        return str(manifest)
    def download_snapshot(repo_id, **kwargs):
        calls.append(('snapshot', repo_id, kwargs))
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download_manifest)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download_snapshot)
    selected = resolve_model_path('owner/collection', subfolder=folder, revision='pinned-commit')
    prefix = folder + '/' if folder else ''
    assert selected == tmp_path / (folder or '')
    assert calls[0] == ('manifest', 'owner/collection', {'filename': prefix + 'pixelone.json', 'revision': 'pinned-commit'})
    assert calls[1] == ('snapshot', 'owner/collection', {
        'revision': 'pinned-commit',
        'allow_patterns': [prefix + 'pixelone.json', prefix + 'config.json', prefix + 'weights-00001.safetensors'],
    })


def test_missing_collection_selection_fails_before_weight_download(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    def missing_manifest(*arguments, **kwargs):
        raise EntryNotFoundError('No manifest')
    def unexpected_snapshot(*arguments, **kwargs):
        raise AssertionError('Weights must not be downloaded without a selected manifest')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', missing_manifest)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', unexpected_snapshot)
    with pytest.raises(ValueError, match='subfolder'):
        resolve_model_path('owner/collection')
