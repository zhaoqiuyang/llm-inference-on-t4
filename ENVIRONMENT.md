# Environment

| Item | Value |
|---|---|
| Platform | Google Colab Pro (cooling and co-tenancy not controllable) |
| GPU | NVIDIA T4, 15 GB, compute capability 7.5, 70 W power limit (min 60 / max 70) |
| Driver / CUDA | 580.82.07 / 13.0 |
| vLLM | 0.34.0 (V1 engine) |
| PyTorch | _fill in: `python -c "import torch; print(torch.__version__)"`_ |
| Models | `Qwen/Qwen2.5-0.5B-Instruct` (fp16), `Qwen/Qwen2.5-7B-Instruct-AWQ` (int4) |

Kernels vLLM selected for the 7B-AWQ runs (from the startup log):

- Linear layers: `MarlinLinearKernel` (w4a16)
- Attention: `TRITON_ATTN` (FlashAttention 2 needs compute capability >= 8.0)
- Sampling: torch fallback (FlashInfer sampler unsupported on 7.5)
- CUDA graphs: on (`FULL_AND_PIECEWISE`)
