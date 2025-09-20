# ggml::torch PyTorch-alignment review & TODOs

## Where the API already mirrors PyTorch
* Tensor method names largely follow PyTorch conventions (`add`, `matmul/mm`, `view`, `reshape`, `sum/mean`, `softmax`, `relu`, etc.), making porting PyTorch model code easier.【F:include/ggml/torch.h†L42-L121】
* The module hierarchy mirrors `torch::nn` with `Module`, `Sequential`, and layer subclasses (`Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `MultiheadAttention`, etc.).【F:include/ggml/torch.h†L288-L490】
* `Tensor::addmm` preserves PyTorch's `beta`/`alpha` semantics, which simplifies loading linear weights from Torch checkpoints.【F:include/ggml/torch.h†L88-L89】【F:src/ggml-torch.cpp†L676-L704】

## Gaps vs. PyTorch that warrant follow-up
* **Dimensionality limits** – reshaping and permutation helpers accept at most 4 dimensions because ggml tensors are restricted accordingly.【F:src/ggml-torch.cpp†L273-L332】 Porting transformer blocks with >4D activations (e.g., grouped convolutions or attention with extra batch dims) would require higher-rank support or explicit tiling logic.
* **Missing common math ops** – the Tensor facade only exposes the operations listed in the header; staples like `exp`, `log`, `tanh`, `sigmoid`, `pow`, `clamp`, `where`, and broadcasting arithmetic with scalars are absent, forcing users back to raw ggml calls or manual graph building.【F:include/ggml/torch.h†L42-L121】 TODO: surface additional unary/binary kernels to close the parity gap with `torch::Tensor`.
* **No dtype/device helpers beyond `to(ggml_type)`** – PyTorch's `.to(device)` and `.cpu()/cuda()` transitions have no analogue; users must juggle `BackendBuffer` manually. Consider adding a backend-aware `to(const Backend&)` that forwards to `Tensor::assign_buffer` when possible.【F:include/ggml/torch.h†L70-L91】【F:src/ggml-torch.cpp†L706-L736】
* **Autograd & training utilities missing** – `nn::Module` only defines `forward`; there is no grad tracking, optimizer hooks, or parameter requires-grad flags, limiting parity with `torch::nn.Module` for training scenarios.【F:include/ggml/torch.h†L296-L305】【F:src/ggml-torch.cpp†L1368-L1460】 TODO: decide whether to expose ggml's backward/optimizer infrastructure or document the expectation of inference-only usage.
* **Parameter initialisation conveniences** – PyTorch initialisers (e.g., `nn::init`) and device-aware parameter buffers are absent. Exposing helper utilities for common initialisation patterns would reduce boilerplate when constructing models from scratch.【F:src/ggml-torch.cpp†L1465-L1656】
* **Graph/scheduler utilities** – the wrapper omits useful scheduler hooks such as `ggml_backend_sched_split_graph` and introspection helpers (`get_n_splits`, `get_n_backends`, etc.), which hampers advanced placement/debugging compared to llama.cpp's usage.【F:include/ggml/torch.h†L213-L257】【F:ggml/include/ggml-backend.h†L320-L337】 TODO: extend `BackendScheduler` to expose these controls.
* **Execution ergonomics** – `Model::generate` always rebuilds schedulers and graphs per token and performs greedy decoding only.【F:src/ggml-torch.cpp†L2025-L2269】 Matching PyTorch's flexible generation APIs would require caching graphs, supporting batched decoding, temperature/logit filtering, etc.

## ggml capabilities not surfaced yet
* **Quantization workflows** – ggml exposes quantization entry points (`ggml_quantize_init/free`, `ggml_quantize_chunk`) for offline or on-the-fly compression, but the wrapper offers no helpers to drive them, limiting usability for custom quantized modules.【F:ggml/include/ggml.h†L2450-L2477】 TODO: provide RAII wrappers or integration paths for quantising `Tensor` data.
* **Backend graph utilities** – ggml can compare backend outputs, copy graphs between devices, and log via callbacks, none of which have equivalents in `ggml::torch` yet.【F:ggml/include/ggml-backend.h†L323-L357】 Document whether these should remain expert-only or expose them for debugging.
* **Advanced ops (convolutions, FFTs, etc.)** – ggml has kernels beyond the current Tensor facade (e.g., convolution, upscaling, embedding bag). Auditing `ggml.h` and incrementally binding frequently used ops would broaden applicability.【F:include/ggml/torch.h†L42-L121】

## Next steps
1. Prioritise adding high-impact PyTorch parity ops (unary math, scalar broadcasting, device transfer) and document behaviour differences where perfect parity is impossible.
2. Decide on the scope of training support: either explicitly document inference-only intent or expose ggml's backward/optimiser APIs through `Tensor`/`Module` extensions.
3. Plan quantization and advanced backend utilities integration so the wrapper can showcase ggml's differentiators instead of just mirroring PyTorch.
