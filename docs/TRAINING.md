[← PixelOne](../README.md) · [Documentation](README.md)

# Training and export

Run commands from the repository root after installation. Configuration paths are relative to the working directory, not to the configuration file. Obtain the corresponding VideoJEV base checkpoint separately and set `base_model` to its local directory or official Hugging Face ID.

## Configuration index

| Stage | Purpose | 4B | 35B-A3B |
| --- | --- | :---: | :---: |
| Search teacher | Train on box-derived and relabeled visited states. | [Config](../configs/teacher_4b.json) | [Config](../configs/teacher_35b.json) |
| 22K student | Initialize from the teacher and learn Top20 + spatial KD. | [Config](../configs/student_22k_4b.json) | [Config](../configs/student_22k_35b.json) |
| PixelOne · 200K | Continue with language LoRA and a frozen visual encoder. | [Config](../configs/pixelone_200k_4b.json) | [Config](../configs/pixelone_200k_35b.json) |
| PixelOne-Pro · 500K | Separately continue with full-parameter updates. | [Config](../configs/pixelone_pro_500k_4b.json) | [Config](../configs/pixelone_pro_500k_35b.json) |

Each continuation requires a teacher cache covering its own dataset. The 200K and 500K branches start from the same trained 22K student and retain its pixel head; Pro is not a continuation of the 200K model. Source counts are listed in the [dataset mixture configuration](../configs/dataset_mixtures.json).

## Stages

1. `teacher_4b.json` / `teacher_35b.json`: 500 updates on box-derived global/local states, then 500 updates using a mixture with states visited by the intermediate teacher. The native decision head and rank-16 language LoRA are trained; vision is frozen. The local crop view is 1,536 pixels, and the initial full view preserves the native screenshot. Partially truncated targets omit visibility loss but retain valid row/column supervision.
2. `student_22k_*.json`: initialize from the trained teacher, create a seeded pixel head, and train for 1,500 updates with Top20 and spatial KD. Set `method` to `gt_top20` for a GT-only run; that mode does not open a teacher cache. `initialization: base` with no `adapter` creates a base-initialized control.
3. `pixelone_200k_*.json`: continue the 22K student using LoRA and a frozen visual encoder.
4. `pixelone_pro_500k_*.json`: separately continue the same 22K student, merge its language LoRA, retain its pixel head, and update all parameters. Only the existing full-attention layers use bidirectional masks. Linear-attention layers keep their native computation; autoregressive KV caching is disabled.

The 22K configurations describe the ancestors of the released continuations, with the seeds and learning rates recorded for those checkpoints. Independent paper ablations use their respective training recipes.

The core package supports T3 teacher training and the `gt_top20` / `pixel_kd_top20` student objectives. Historical search schedules, single-winner/Top100 variants and hard/region-KD ablations are not exposed by this launcher. Both the reference and vectorized spatial KD implementations remain available through the existing recipe setting.

## Teacher caches

`cache-teacher` uses the data path in its supplied teacher configuration. To cache a continuation dataset, copy the teacher configuration and change `data`, `image_root`, and `expected_targets` to that continuation. Use the same final teacher adapter at a given model scale. Cache files record probabilities, crop bounds, instruction/box identity and screenshot hashes, without dense per-pixel targets. A SQLite byte-offset index is created alongside the JSONL.

Set each student configuration's `teacher_cache` and `teacher_index` to the resulting pair. Opening a cache includes a full checksum scan. Retrieved records are also checked against the target's image, instruction and box identity.

## Distributed training

Use `torchrun`. `global_batch` must be divisible by `WORLD_SIZE * micro_batch`; gradient accumulation maintains the configured global batch on smaller GPU counts. Pro training uses FSDP, with hybrid sharding when requested across multiple nodes. LoRA/teacher training uses DDP. Multi-node runs require shared access to inputs, output checkpoints and visited-state files.

The released Pro recipes use global batch 256 and one pass over 500K targets (1,954 updates, including the final padded stream batch). Their native kernels and grouped MoE implementation require the reference-compatible CUDA environment. Memory consumption depends strongly on screenshot resolution; model-weight size alone is not the required device memory.

## Resume

```bash
torchrun --nproc_per_node=8 -m pixelone.cli train \
  --config configs/pixelone_pro_500k_4b.json \
  --resume outputs/pixelone_pro_500k_4b/step_001000
```

Checkpoints contain model and optimizer state, the next sampling position, recipe/data identity and per-rank PyTorch RNG state. Resuming validates the recipe, dataset/cache/adapter identities, processor assets and world size. Do not edit or reuse an incomplete checkpoint. Mixed-phase teacher checkpoints also bind `visited_states.json`; resuming into another output directory copies and verifies the original file rather than rerunning search with a later teacher.

## Export Pro weights

```bash
pixelone export --checkpoint outputs/pixelone_pro_500k_4b/step_001954 \
  --output outputs/PixelOne-Pro-4B
```

Export streams only merged model tensors from the distributed checkpoint into safetensors shards. Optimizer/RNG state and teacher caches are not exported. Processor assets are checked against training metadata. `--precision training_cuda` preserves the reference mixed-precision partition; `float32` is intended for small CPU parity fixtures and increases memory use substantially on real models.

The export is directly loadable by `PixelOnePredictor`. Adapter checkpoints from teacher/LoRA training are saved separately as `adapter.safetensors` for the next stage; they are not complete Pro inference models.
