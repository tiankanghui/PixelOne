[← PixelOne](../README.md) · [Documentation](README.md)

# Inference guide

## Model packages

Both Pro variants are complete, full-parameter 500K continuations. Inference does not require a base-model download, a separate LoRA adapter, a teacher checkpoint, or a teacher cache.

| Model | Weight files | Approximate weight size | Mean accuracy |
| --- | ---: | ---: | ---: |
| PixelOne-Pro-4B | 3 safetensors shards | 10.3 GB | 80.62% |
| PixelOne-Pro-35B-A3B | 16 safetensors shards | 68.7 GB | 83.77% |

The weights are losslessly repacked with a 5 GB target shard size. Tensor values and the original BF16/FP32 partition are unchanged. Download the updated `pixelone.json` and `model.safetensors.index.json` together with the new shards; do not mix old manifests with repacked weights.

The previous shard files were permanently retired from Hub history. Older pinned revisions may no longer load; pin the repacked release for new deployments.

These sizes exclude activations and runtime overhead. Required GPU memory increases with screenshot resolution. Both models are stored in [KangHui2200/PixelOne](https://huggingface.co/KangHui2200/PixelOne), under `PixelOne-Pro-4B` and `PixelOne-Pro-35B-A3B`. Authenticate with Hugging Face if the repository is private.

Each model directory contains the weights, tokenizer/processor assets, `pixelone.json`, a tensor index, a model card and evaluation counts. Both models share the inference-only `runtime/` at the Hugging Face repository root. Use `PixelOnePredictor`, not `AutoModelForCausalLM.generate` or a generic VideoJEV loader.

### Benchmark results

| Model | ScreenSpot | ScreenSpot V2 | ScreenSpot Pro | OSWorld-G |
| --- | ---: | ---: | ---: | ---: |
| PixelOne-Pro-4B | 90.33 | 92.85 | 68.12 | 71.18 |
| PixelOne-Pro-35B-A3B | 91.67 | 94.50 | 72.04 | 76.86 |

Values are point-in-box accuracy (%) on complete benchmark sets. The four-benchmark mean weights the benchmarks equally. The model packages' `evaluation.json` files retain hit counts and unrounded accuracies. Latency values come from the paper's H200 evaluation records.

## Installation

Use Python 3.12 and a CUDA-enabled PyTorch 2.11 installation. From the code repository:

```bash
python -m pip install -e .
python -m pip install flash-linear-attention==0.4.2 causal-conv1d==1.7.0 --no-build-isolation
```

For inference only, install the shared Hugging Face runtime instead of the full research package:

```bash
python -m pip install -U huggingface_hub
hf download KangHui2200/PixelOne --include 'runtime/*' --local-dir ./PixelOne
python -m pip install ./PixelOne/runtime
python -m pip install flash-linear-attention==0.4.2 causal-conv1d==1.7.0 --no-build-isolation
```

Use a separate environment for each package; they share the same package name. For reproducibility, pin the runtime download and model loader to the same Hugging Face commit.

| Component | Reference version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.11.0 |
| Transformers | 5.6.2 |
| Triton | 3.7.0 |
| FLA | 0.4.2 |
| causal-conv1d | 1.7.0 |

Native kernels must match the CUDA/PyTorch environment. Missing kernels cause an explicit error, not a silent fallback. `--kernel-backend torch` selects the reference PyTorch implementation for diagnostics; it is not the reported latency configuration.

## Command line

```bash
pixelone infer \
  --model /path/to/PixelOne-Pro-4B \
  --image screenshot.png \
  --instruction "Click the search button" \
  --device cuda:0
```

The command prints JSON. `point` is the predicted `[x, y]` in original screenshot pixels; `token_index` and `pixel_index` identify the two readout decisions. The image metadata records the original dimensions and padded canvas.

## Python API

```python
from PIL import Image
from pixelone import PixelOnePredictor

predictor = PixelOnePredictor.from_pretrained(
    "/path/to/PixelOne-Pro-4B",
    device="cuda:0",
)
with Image.open("screenshot.png") as image:
    result = predictor.predict(image, "Click the search button")
print(result["point"])
```

A Hugging Face model ID can replace the local path. For this collection, select a subfolder:

```python
predictor = PixelOnePredictor.from_pretrained(
    "KangHui2200/PixelOne",
    subfolder="PixelOne-Pro-4B",
    device="cuda:0",
)
```

The CLI equivalent is `--model KangHui2200/PixelOne --subfolder PixelOne-Pro-4B`. The loader reads the selected manifest first and downloads only its required files, not the other model. Pin a `revision` for reproducible deployments. Loading verifies weight checksums by default, which scans the shards before inference; keep the predictor loaded when processing multiple targets.

## Input and output contract

- Screenshots retain their original resolution and are padded only on the right and bottom to stride 32.
- Predictions use half-pixel centers in original-image coordinates. Padding is excluded from the final pixel argmax.
- The decoder commits to a visual token before choosing a pixel within it.
- The model uses one visual encoding and one language-model forward, without spatial search or autoregressive coordinate decoding.
- Empty instructions, model control tokens in instructions, or inputs exceeding the context budget are rejected explicitly.

> [!IMPORTANT]
> The released mixed-precision weights require CUDA. Do not call `.half()` or cast the entire model to BF16: some parameters intentionally remain FP32. Use the loader's native precision policy.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Native Qwen kernels unavailable | Install the pinned FLA/causal-conv1d versions compatible with your PyTorch and CUDA environment. |
| CUDA out of memory | Allow memory for both weights and resolution-dependent activations; the file size is not the total GPU memory requirement. |
| Context budget exceeded | The runtime never silently downsamples. Preprocess explicitly only if a changed input-resolution policy is acceptable for your use case. |
| Missing or mismatched weights | Restore the complete model package, including its manifest and all shards; do not mix files from different checkpoints. |
| Generic model loader fails | Install this repository or the package's `runtime/`, then use `PixelOnePredictor`. |

The predictor returns coordinates only. Downstream applications should validate actions and enforce permissions before executing clicks. See [reproducibility and validation](REPRODUCIBILITY.md) for the checks completed during release preparation.
