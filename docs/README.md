[← PixelOne](../README.md)

# Documentation

Choose the shortest path to your task.

| I want to… | Start here |
| --- | --- |
| Predict a click from a screenshot | [Inference guide](INFERENCE.md) |
| Choose a model or check hardware requirements | [Model packages](INFERENCE.md#model-packages) |
| Train a teacher or continue a student | [Training and export](TRAINING.md) |
| Format data and understand the source mixtures | [Dataset configuration](DATASETS.md) |
| Read or extend the implementation | [Code guide](CODE_GUIDE.md) |
| Check numerical validation and reproduction scope | [Reproducibility](REPRODUCIBILITY.md) |
| Read the method and experiments | [Paper on arXiv](https://arxiv.org/abs/2611.30235) |
| Watch real browser clicks at 2K and 4K | [Live demos](DEMOS.md) |
| Review upstream attribution | [Third-party notices](../THIRD_PARTY_NOTICES.md) |

## Training path

**Search teacher → cached spatial distributions → 22K student → separate continuations**

The 200K PixelOne and 500K PixelOne-Pro models branch independently from the same trained 22K student. The [training guide](TRAINING.md#stages) explains the stages, caches, distributed launch, resume and export.

## Testing

From the repository root, install the test dependencies and run the unit tests:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

To also run tiny dense/MoE integration tests, provide the tokenizer/processor assets from a Pro package. These tests do not load the full model weights:

```bash
PIXELONE_TEST_PROCESSOR=/path/to/PixelOne-Pro-4B python -m pytest -q
```

Tests cover teacher search, readout round trips, training resume, weight export, kernel/precision policies and selective Hugging Face subfolder loading. Tests use local assets without automatic downloads. See [Live demos](DEMOS.md) for the GPU recordings and [Validation](REPRODUCIBILITY.md#checks-performed-in-the-release-workspace) for test coverage and benchmark provenance.

## Repository map

The [code guide](CODE_GUIDE.md) maps commands to implementation entry points and gives a reading order for the model and losses.

```text
configs/                       Training recipes and dataset mixtures
docs/                          Guides, paper and illustrations
src/pixelone/                  Training, inference, data and checkpoint workflows
  model.py                     Shared teacher/student model and both prediction heads
  search.py, states.py         T3 search and teacher supervision
  losses.py, distillation.py   Top20 box supervision and observed-support spatial KD
  runtime.py                   Attention, native kernels and inference precision
src/videojev_hf/               Upstream Qwen / VideoJEV architecture
tests/                         Numerical and tiny-model integration tests
third_party/                   Upstream licenses and notices
```

## Packages

This repository contains training and inference code, configurations and tests. Weights and the inference-only runtime are in the [model collection](https://huggingface.co/KangHui2200/PixelOne). Exact training annotations and image reconstruction instructions are in [PixelOne-Data](https://huggingface.co/datasets/KangHui2200/PixelOne-Data).

The training launcher supports T3 teacher training and the `gt_top20` / `pixel_kd_top20` student objectives.

Licensing and upstream attribution are documented in [Third-party notices](../THIRD_PARTY_NOTICES.md).
