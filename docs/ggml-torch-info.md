# ggml::torch API overview

## Purpose and design
The C++ wrapper in `ggml::torch` mirrors a subset of PyTorch's tensor and module APIs on top of ggml primitives. `Tensor` owns a raw `ggml_tensor *` plus the originating `Context`, providing arithmetic, reduction, view, and layout helpers that always operate within the same context.【F:include/ggml/torch.h†L28-L145】【F:src/ggml-torch.cpp†L132-L188】 Neural-network layers derive from `nn::Module`, which mirrors `torch::nn::Module` by registering parameters and buffers tied to the shared ggml context.【F:include/ggml/torch.h†L288-L321】【F:src/ggml-torch.cpp†L1465-L1475】 Backend wrappers expose ggml backends, buffers, and schedulers while keeping ownership and allocation rules explicit.【F:include/ggml/torch.h†L147-L257】【F:src/ggml-torch.cpp†L859-L915】【F:src/ggml-torch.cpp†L918-L1219】 The `Model` helper adds GGUF weight loading and a simple token-generation loop that builds graphs on demand and dispatches them through a backend scheduler.【F:include/ggml/torch.h†L491-L545】【F:src/ggml-torch.cpp†L1792-L2269】

## Context management
Create contexts through `Context::create`, which wraps `ggml_init` and guarantees `shared_ptr` ownership.【F:include/ggml/torch.h†L259-L284】【F:src/ggml-torch.cpp†L128-L205】【F:src/ggml-torch.cpp†L1215-L1244】 Contexts allocate tensors (`new_tensor`, `new_f32`, `new_i32`) and can bulk-allocate all context tensors onto a `Backend` or buffer type via `allocate_tensors` helpers.【F:include/ggml/torch.h†L270-L278】【F:src/ggml-torch.cpp†L37-L213】【F:src/ggml-torch.cpp†L201-L214】【F:src/ggml-torch.cpp†L1246-L1279】 Because contexts are shared, every `Tensor` checks that operands come from the same context before running ggml kernels.【F:src/ggml-torch.cpp†L167-L188】

```cpp
#include "ggml/torch.h"

const ggml_init_params params{
    /*.mem_size   =*/ 1ULL << 26,
    /*.mem_buffer =*/ nullptr,
    /*.no_alloc   =*/ false,
};
auto ctx = ggml::torch::Context::create(params);

auto a = ctx->new_tensor(GGML_TYPE_F32, {128, 64});
auto b = ctx->new_tensor(GGML_TYPE_F32, {64, 32});
auto c = a.matmul(b); // shares ctx with a/b automatically
```

## Tensor API
### Shape and layout
* `sizes`, `size`, `dim`, and `numel` provide introspection of up to four dimensions (ggml's maximum).【F:include/ggml/torch.h†L37-L40】【F:src/ggml-torch.cpp†L137-L165】【F:src/ggml-torch.cpp†L37-L54】
* Views and reshaping include `reshape`, `view`, `view_as`, and `flatten`, all limited to 1–4D shapes enforced at runtime.【F:include/ggml/torch.h†L63-L69】【F:src/ggml-torch.cpp†L199-L214】【F:src/ggml-torch.cpp†L246-L294】
* Indexing helpers mirror PyTorch semantics: `unsqueeze`, `squeeze`, `narrow`, `select`, `chunk`, `split`, and `split_with_sizes` handle dimension normalization and slicing bounds, while `repeat`, `repeat_like`, and `expand` build broadcast-compatible references.【F:include/ggml/torch.h†L74-L88】【F:src/ggml-torch.cpp†L630-L756】【F:src/ggml-torch.cpp†L696-L760】

### Arithmetic, reductions, and activations
* Elementwise operations (`add`, `sub`, `mul`, `div`, `scale`, `neg`) and matrix multiplication (`matmul`/`mm`, `addmm`) forward directly to ggml kernels after verifying context compatibility.【F:include/ggml/torch.h†L42-L49】【F:src/ggml-torch.cpp†L296-L334】【F:src/ggml-torch.cpp†L600-L611】
* Reductions (`sum`, `mean`) and activations (`softmax`, `silu`, `gelu`, `relu`) implement the expected dimension semantics via ggml functions, including a shared helper to keep reduced dimensions when requested.【F:include/ggml/torch.h†L50-L58】【F:src/ggml-torch.cpp†L232-L383】
* Normalization utilities (`layer_norm`, `rms_norm`, `diag_mask_inf`, `diag_mask_zero`) expose ggml's norm and masking operators.【F:include/ggml/torch.h†L57-L60】【F:src/ggml-torch.cpp†L375-L393】

### Layout transforms and conversion
* `transpose` and `permute` accept negative indices and deduplicate axes, while `contiguous`, `clone`, and `to` create explicit buffers with new layout or dtype (`ggml_cast`).【F:include/ggml/torch.h†L62-L73】【F:src/ggml-torch.cpp†L395-L470】【F:src/ggml-torch.cpp†L453-L470】
* `index_select` maps to `ggml_get_rows`, enabling embedding-style lookups.【F:include/ggml/torch.h†L88-L89】【F:src/ggml-torch.cpp†L688-L698】
* `assign_buffer` couples tensors to an allocated `BackendBuffer`, prohibiting views and double assignment.【F:include/ggml/torch.h†L91-L91】【F:src/ggml-torch.cpp†L613-L628】

### Attention-specific helpers
* `RopeConfig` bundles rotary-embedding parameters used by `rope`, which wraps `ggml_rope_ext` with optional frequency factors.【F:include/ggml/torch.h†L93-L110】【F:src/ggml-torch.cpp†L781-L806】
* `flash_attention` exposes ggml's fused attention kernel with optional additive mask, scale, bias, and logit soft-capping controls.【F:include/ggml/torch.h†L111-L121】【F:src/ggml-torch.cpp†L808-L833】
* Static `concat` joins tensors along any normalized axis by repeatedly invoking `ggml_concat`.【F:include/ggml/torch.h†L123-L124】【F:src/ggml-torch.cpp†L835-L857】

```cpp
auto logits = query.flash_attention(key, value, /*scale=*/1.0f / std::sqrt(head_dim));
auto masked = logits.diag_mask_inf(n_past).softmax();
auto pooled = masked.sum(/*dim=*/1, /*keepdim=*/false);
```

## Backend integration
* `BackendBuffer` wraps `ggml_backend_buffer_t` with RAII, allocation reset, tensor initialization, and metadata queries such as alignment and host/device placement.【F:include/ggml/torch.h†L147-L175】【F:src/ggml-torch.cpp†L859-L915】
* `Backend` constructs CPU/GPU handles or loads backends by type/name, and can allocate raw buffers or pack tensor allocations in the backend's preferred buffer type with optional usage hints.【F:include/ggml/torch.h†L177-L211】【F:src/ggml-torch.cpp†L918-L1107】
* `BackendScheduler` wraps `ggml_backend_sched_t`, exposing reservation, allocation, synchronous/asynchronous compute, tensor placement, and evaluation callbacks for multi-backend execution.【F:include/ggml/torch.h†L213-L257】【F:src/ggml-torch.cpp†L1136-L1239】

```cpp
auto cpu = ggml::torch::Backend::cpu(4);
auto buffer = cpu.alloc_tensors({weight, bias}, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
weight.assign_buffer(buffer);
buffer.set_usage(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
```

## Module library
* `nn::Module` stores parameters, buffers, and submodules keyed by string, and recursively enumerates them for serialization or optimizer setup.【F:include/ggml/torch.h†L288-L321】【F:src/ggml-torch.cpp†L1408-L1461】
* `Linear` builds weight (shape `[in_features, out_features]`) and optional bias tensors, using `matmul` plus broadcasted bias addition in `forward`.【F:include/ggml/torch.h†L326-L342】【F:src/ggml-torch.cpp†L1465-L1491】
* `RotaryEmbedding` stores a defaulted `RopeConfig` and exposes `apply` helpers that rotate query/key tensors given positions and optional frequency factors.【F:include/ggml/torch.h†L344-L363】【F:src/ggml-torch.cpp†L1493-L1534】
* `FeedForward` wires up gated or ungated MLP blocks using shared `Linear` instances and SiLU activation.【F:include/ggml/torch.h†L365-L384】【F:src/ggml-torch.cpp†L1618-L1655】
* `MultiheadAttention` manages projection layers, optional rotary embeddings, reshape/permute patterns, and fused flash-attention with optional mask injection.【F:include/ggml/torch.h†L386-L420】【F:src/ggml-torch.cpp†L1658-L1741】
* `Embedding`, `LayerNorm`, `RMSNorm`, and `Sequential` provide common building blocks that mirror PyTorch signatures and semantics.【F:include/ggml/torch.h†L422-L490】【F:src/ggml-torch.cpp†L1536-L1771】

```cpp
auto mha = std::make_shared<ggml::torch::MultiheadAttention>(ctx, /*embed_dim=*/512, /*num_heads=*/8);
auto ff  = std::make_shared<ggml::torch::FeedForward>(ctx, 512, 2048, /*gated=*/true);

ggml::torch::Sequential block(ctx);
block.append("mha", mha).append("ff", ff);

auto hidden = block.forward(hidden_states);
```

## Model utilities
`Model` extends `nn::Module` with:
* A `ConfigMap` populated from GGUF metadata (supports scalars and arrays of multiple primitive types).【F:include/ggml/torch.h†L493-L518】【F:src/ggml-torch.cpp†L1792-L2003】
* Backend-aware weight loading that groups tensors per backend, allocates buffers with `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, and streams tensor bytes from the GGUF file into backend memory.【F:src/ggml-torch.cpp†L1906-L2021】
* Execution backend preparation that always includes a CPU backend, inspects parameter buffers for device types, and instantiates backends for each unique buffer type.【F:src/ggml-torch.cpp†L2024-L2068】
* Placement bookkeeping and scheduler creation to bind tensors to backends before running a graph.【F:src/ggml-torch.cpp†L2077-L2140】
* A greedy `generate` loop that rebuilds a graph around the latest logits, reserves/allocates backend resources each token, uploads prompt tokens, executes the scheduler synchronously, and samples the next token from the final logit row.【F:src/ggml-torch.cpp†L2200-L2269】

```cpp
std::vector<int> prompt = {1, 42, 128};
model.load_weights_from_gguf("my-model.gguf");
auto completion = model.generate(prompt, /*n_tokens=*/32);
```

