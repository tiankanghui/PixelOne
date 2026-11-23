# Third-party attribution

- VideoJEV model code and native decision interface are retained in `src/videojev_hf/`. The supplied upstream Apache-2.0 license and notice are in `third_party/VideoJev-LICENSE` and `third_party/VideoJev-NOTICE`.
- Qwen3.5-4B and Qwen3.6-35B-A3B provide the underlying dense/MoE architectures and base weights. Their upstream licenses remain applicable to derived models.
- Transformers, PyTorch, Accelerate, safetensors and optional FLA/causal-conv1d kernels are installed dependencies, not vendored replacements. Their respective licenses apply.
- DejaVuSans is included only for the teacher's labeled-grid rendering. Its license is bundled beside the font in `src/pixelone/assets/DejaVuSans-LICENSE.txt`.
- An optional historical XTuner expert path remains in the inherited VideoJEV source. The published Pro-35B recipe uses PyTorch `grouped_mm`, not that optional XTuner path.

The GitHub source license still requires maintainer confirmation. The Hugging Face collection retains the MIT license metadata already configured by its owner; upstream licenses and notices remain applicable. This notice preserves attribution and does not itself grant a new license.
