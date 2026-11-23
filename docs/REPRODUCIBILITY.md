[← PixelOne](../README.md) · [Documentation](README.md)

# Reproducibility and validation

The released Pro safetensors are losslessly repacked from the final inference exports, without quantization or tensor modification. Every tensor's name, shape, dtype and byte-level SHA256 match the original release. Their source evaluation records match the paper's complete benchmark counts and hit counts. `evaluation.json` in each model package stores these counts; `pixelone.json` binds the inference recipe, tensor shapes/dtypes and updated file hashes.

The runtime preserves native-resolution padding, query and visual-token readout, Top20 and observed-support KD definitions, coarse-first decoding, the original valid-image mask, bidirectional existing full-attention layers, FP32 router arithmetic, and the reference mixed-precision partition. A common CLI provides training, inference, data preparation and checkpoint handling.

Exact replay of a run requires its sampled target IDs, source-image grouping, teacher caches, software/kernel versions, GPU topology and data ordering. The public launcher uses a deterministic sample stream and gradient accumulation. Historical bucketing and prefetch settings are separate from this launcher. The default 22K configurations follow the checkpoint recipes used by the released continuations.

## Checks performed in the release workspace

- Unit tests: original-pixel geometry and padding, token-first decoding, truncated visibility labels, Top20 gradients, GT/KD readout separation, vectorized KD, cache identity, and input-path validation.
- Tiny dense/MoE CPU tests: actual three-call/five-encoding T3 search and bit-identical direct logits after full-weight save/load.
- Tiny CPU training: LoRA and full-parameter checkpoint resume, plus sharded checkpoint export and reload.
- Core-package cleanup: separate-process dense/MoE comparisons against the previous implementation preserve teacher loss/gradients, T3 traces, supported student losses/gradients and direct predictions exactly on CPU fixtures. Both released model configurations retain the same tensor keys and shapes.
- Collection loading: explicit local/Hugging Face subfolder selection, safe paths, and downloading only the selected manifest's files.
- Full released weights: original shard SHA256 verification, per-tensor byte equality after repacking, updated shard SHA256 verification, and unchanged source-evaluation identity.

Live H200 recordings cover Pro-4B in the 2560 × 1440 vector editor and 3840 × 2160 spreadsheet, and Pro-35B-A3B in the 2560 × 1440 Three Suns game. These use native FLA and causal-conv1d kernels. Model inputs exclude display overlays, and clicks are checked against application states. Recording details are in [Live demos](DEMOS.md).

Benchmark scores and H200 timings come from the paper's evaluation records. Release validation includes CPU tests and selected functional GPU demos. Benchmark timing and multi-GPU FSDP/NCCL training are outside this validation scope.

[Code](https://github.com/tiankanghui/PixelOne) · [Models](https://huggingface.co/KangHui2200/PixelOne) · [Licensing and attribution](../THIRD_PARTY_NOTICES.md)
