[← PixelOne](../README.md) · [Documentation](README.md)

# Code guide

Start with the entry point for your task, then follow the reading order below to understand the model, supervision and training loop.

## Repository layout

```text
configs/                 Teacher, student and continuation recipes; source mixtures
src/pixelone/            PixelOne training, inference, supervision and data handling
src/videojev_hf/         Required Qwen / VideoJEV architecture and processor support
tests/                   Unit tests and tiny dense/MoE integration tests
docs/                    Guides, paper, recordings and case illustrations
third_party/             Upstream licenses and attribution
```

`pixelone` contains the project workflows and supervision code. `videojev_hf` provides the backbone architecture and processor support. Model weights and training annotations are hosted in the linked Hugging Face repositories.

## Start from a workflow

All commands enter through [`cli.py`](../src/pixelone/cli.py). The CLI imports model dependencies only when a command needs them.

| Task | Command | Implementation entry point |
| --- | --- | --- |
| Predict a click | `pixelone infer` | [`PixelOnePredictor`](../src/pixelone/inference.py) |
| Train or resume a run | `pixelone train` | [`run_training`](../src/pixelone/training.py) |
| Validate grounding annotations | `pixelone prepare-data` | [`prepare_dataset`](../src/pixelone/data.py) |
| Save teacher search traces | `pixelone cache-teacher` | [`SearchTeacher` and `cache_teacher`](../src/pixelone/teacher.py) |
| Export Pro weights | `pixelone export` | [`export_checkpoint`](../src/pixelone/export.py) |

Runnable commands and environment requirements are in the [inference](INFERENCE.md) and [training](TRAINING.md) guides.

### Inference

```text
PixelOnePredictor.from_pretrained
  → select and verify a model bundle
  → load FullWeightStudent and apply its runtime policy

PixelOnePredictor.predict
  → prepare the original-resolution screenshot and instruction
  → one student forward
  → decode_click: select a token, then a valid pixel within it
```

Read [`inference.py`](../src/pixelone/inference.py), then [`inputs.py`](../src/pixelone/inputs.py), [`model.py`](../src/pixelone/model.py), and [`geometry.py`](../src/pixelone/geometry.py). `runtime.py` controls the attention, kernel and precision policies. The public import remains `from pixelone import PixelOnePredictor`.

### Training

[`training.py`](../src/pixelone/training.py) owns run setup, sampling, optimization, resume and checkpoint saving. Its branches follow the selected recipe:

- **Teacher:** sample box-derived states, train the native decision interface, collect visited states, then continue with mixed supervision. See `states.py`, `teacher.py` and `search.py`.
- **22K / 200K student:** initialize the student, read GT and optional teacher support, and optimize language LoRA plus the readout heads. See `model.py`, `losses.py` and `distillation.py`.
- **500K Pro:** retain the trained pixel head, merge the language LoRA, and enable full-parameter training. `training.py` uses the batch wrapper, optimizer groups and FSDP setup from `full_training.py`.

The 200K and 500K configurations branch independently from the same 22K student. The [recipe table](TRAINING.md#configuration-index) lists the configuration files.

## Read the method implementation

| Order | File | What to look for |
| ---: | --- | --- |
| 1 | [`geometry.py`](../src/pixelone/geometry.py) | `Layout`, the row-major token grid, pixel centers and padding mask. |
| 2 | [`model.py`](../src/pixelone/model.py) | The shared native decision head, `FineReadout`, token routing, and teacher/student initialization. |
| 3 | [`losses.py`](../src/pixelone/losses.py) | Top20 winner loss, box supervision, and the GT token-average denominator. |
| 4 | [`distillation.py`](../src/pixelone/distillation.py) | Teacher leaf projection and observed-support KD. Read `kd_loss` before its vectorized equivalent `pixel_kd_loss`. |
| 5 | [`search.py`](../src/pixelone/search.py) | Row/column cell scores and the three-round, two-candidate T3 search. |
| 6 | [`training.py`](../src/pixelone/training.py) | How the model, objectives, sampling and checkpoints are used together. |

The cosine option scorer itself is `CosineDecisionHead` in [`videojev_hf/modeling_videojev.py`](../src/videojev_hf/modeling_videojev.py).

### Supporting modules

| Files | Responsibility |
| --- | --- |
| `batching.py`, `processor.py` | Pack questions and connect the processor to the native decision interface. |
| `states.py`, `views.py` | Build teacher supervision and render its global/local search views. |
| `teacher.py` | Run the trained teacher and write cached search traces. |
| `data.py`, `sampling.py` | Validate annotations, look up teacher records and preserve deterministic sample order. |
| `checkpoint.py`, `export.py` | Resume complete training state or export model-only inference bundles. |
| `io.py` | Validate bundle paths, compute checksums and write metadata atomically. |
| `runtime.py`, `full_training.py` | Apply kernel/attention/precision policies and distributed training setup. |

## Implementation notes

- Coordinates refer to original screenshot pixels. The valid-pixel mask excludes padding. Inputs that exceed the context budget raise an error.
- Inference selects the highest-scoring token, then the highest-scoring valid pixel within it.
- `native.head` is the shared decision head, and `spatial` is the pixel head. Saved tensor keys use these names.
- GT routing and the pixel-loss denominator are independent of additional teacher-observed readouts. Tokens fully covered by the target still contribute zero to that average.
- KD uses observed support and a correct-teacher-click gate, with temperature scaling and original token-mass weights.
- Pro uses bidirectional masks in its existing full-attention layers and retains native linear attention.
- Training checkpoints include model, optimizer and RNG state for resume. Inference bundles contain model tensors, processor assets and a manifest.

## Tests to read alongside the code

- [`test_public_core.py`](../tests/test_public_core.py): geometry, decoding, Top20 gradients, GT/KD separation and data identity.
- [`test_core_workflows.py`](../tests/test_core_workflows.py): CLI coverage, KD equivalence, runtime restoration and precision handling.
- [`test_hub_collection.py`](../tests/test_hub_collection.py): safe local/Hub subfolder selection and selective downloads.
- [`test_model_integration.py`](../tests/test_model_integration.py): tiny dense/MoE forward passes, T3 call counts, resume and export.

See [Testing](README.md#testing) for commands. Integration tests use local processor assets and small CPU models. The [validation guide](REPRODUCIBILITY.md) describes test coverage and benchmark provenance.
