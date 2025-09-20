# ggml::torch C++ API overview

## Design summary
The wrapper exposes ggml primitives through a `ggml::torch` namespace that mirrors portions of PyTorch's tensor and module APIs while keeping ggml's explicit context, backend, and buffer management. A `Context` owns the underlying `ggml_context` and creates tensors, modules keep shared ownership of that context, and backends/schedulers manage device execution and memory placement.【F:include/ggml/torch.h†L20-L545】【F:src/ggml-torch.cpp†L962-L2270】 Tensors always carry a pointer back to their originating `Context`; most operations validate that operands share the same context before calling into ggml kernels.【F:include/ggml/torch.h†L32-L125】【F:src/ggml-torch.cpp†L186-L314】

## Core tensor API

### Construction and context access
* `Context::create` builds a reference-counted context from `ggml_init_params`; `Context::new_tensor/new_f32/new_i32` allocate tensors of specific shapes and dtypes; `Context::wrap` turns a raw `ggml_tensor*` into a managed `Tensor`.【F:include/ggml/torch.h†L259-L284】【F:src/ggml-torch.cpp†L1285-L1342】
* Each `Tensor` knows its context (`Tensor::context`), raw pointer (`Tensor::raw`), and whether it is defined (`Tensor::defined`).【F:include/ggml/torch.h†L32-L40】【F:src/ggml-torch.cpp†L186-L239】 All tensor operations call `require_defined` and `require_same_context`, throwing if invariants are broken.【F:src/ggml-torch.cpp†L186-L248】

```cpp
using namespace ggml::torch;

ggml_init_params params{/*.mem_size=*/1ull << 20, /*.mem_buffer=*/nullptr, /*.no_alloc=*/false};
auto ctx = Context::create(params);
Tensor a = ctx->new_tensor(GGML_TYPE_F32, {4, 4});
Tensor b = ctx->new_tensor(GGML_TYPE_F32, {4, 4});
Tensor c = a.add(b).relu();
```

### Shape inspection and reduction helpers
* `Tensor::sizes`, `size`, `dim`, and `numel` query tensor metadata, normalising ggml's `n_dims` semantics.【F:include/ggml/torch.h†L37-L40】【F:src/ggml-torch.cpp†L214-L244】
* Reductions include scalar sum/mean and dimension-aware `sum/mean(dim, keepdim)` implemented via `reduce_rows_like`, which permutes axes so ggml's row-reduction kernels can be reused.【F:include/ggml/torch.h†L50-L52】【F:src/ggml-torch.cpp†L249-L331】

### Arithmetic, activation, and normalisation
* Elementwise operators (`add`, `sub`, `mul`, `div`, `scale`, `neg`) forward to ggml kernels after context validation.【F:include/ggml/torch.h†L42-L49】【F:src/ggml-torch.cpp†L333-L386】
* Matrix multiplication uses `ggml_mul_mat` (`Tensor::matmul/mm`).【F:include/ggml/torch.h†L43-L44】【F:src/ggml-torch.cpp†L339-L348】
* Activation and norm helpers (`softmax`, `silu`, `gelu`, `relu`, `layer_norm`, `rms_norm`, `diag_mask_*`) wrap the corresponding ggml ops, including the quick GELU branch selection.【F:include/ggml/torch.h†L53-L60】【F:src/ggml-torch.cpp†L388-L446】
* `addmm` mimics PyTorch by optionally scaling the base tensor (`beta`) and product (`alpha`) before summation.【F:include/ggml/torch.h†L88-L89】【F:src/ggml-torch.cpp†L676-L704】

### View, reshape, and indexing utilities
* Permutation and reshape APIs accept up to 4 dimensions (ggml limitation) and reuse helper arrays for axis management.【F:include/ggml/torch.h†L62-L80】【F:src/ggml-torch.cpp†L250-L332】【F:src/ggml-torch.cpp†L447-L616】
* `flatten`, `contiguous`, `clone`, `to`, and `view_as` create new layout-compatible tensors by calling the appropriate ggml reshape/view helpers.【F:include/ggml/torch.h†L68-L72】【F:src/ggml-torch.cpp†L447-L616】
* Indexing helpers (`narrow`, `select`, `chunk`, `split`, `split_with_sizes`, `index_select`) use ggml views and row-gather kernels to avoid data copies.【F:include/ggml/torch.h†L76-L80】【F:src/ggml-torch.cpp†L617-L778】

### Broadcasting, repetition, and expansion
* `repeat`/`repeat_like`/`expand` create repeat-reference tensors by allocating a reference shape tensor and invoking `ggml_repeat`.【F:include/ggml/torch.h†L82-L87】【F:src/ggml-torch.cpp†L647-L742】
* `expand_as` forwards to `expand` with the target tensor's sizes.【F:include/ggml/torch.h†L86-L87】【F:src/ggml-torch.cpp†L705-L742】

### Rotary embeddings and flash attention
* `Tensor::rope` exposes ggml's extended rotary positional encoding, accepting optional frequency factors and the detailed `RopeConfig` structure (base, scale, beta, etc.).【F:include/ggml/torch.h†L93-L109】【F:src/ggml-torch.cpp†L781-L806】
* `Tensor::flash_attention` provides both masked and unmasked variants using `ggml_flash_attn_ext`, including optional bias softcapping parameters.【F:include/ggml/torch.h†L111-L121】【F:src/ggml-torch.cpp†L808-L833】

### Buffer assignment and concatenation
* `Tensor::assign_buffer` binds a tensor to a `BackendBuffer`, validating that the tensor is not a view and that buffers are consistent before calling `ggml_backend_buffer_init_tensor`.【F:include/ggml/torch.h†L91-L91】【F:src/ggml-torch.cpp†L706-L736】
* Static `concat` helpers append tensors along a dimension, chaining ggml's `ggml_concat` for multi-input cases.【F:include/ggml/torch.h†L123-L124】【F:src/ggml-torch.cpp†L835-L856】

## Backend and buffer management

### BackendBuffer
A lightweight RAII wrapper over `ggml_backend_buffer_t` that provides reset, `init_tensor`, usage flags, and metadata queries (size, alignment, `is_host`). It forbids copying and owns the buffer lifetime.【F:include/ggml/torch.h†L147-L175】【F:src/ggml-torch.cpp†L859-L931】

### Backend
Represents a device backend and manages ownership of the underlying `ggml_backend_t`. Factory helpers construct CPU, GPU, by-type, or by-name backends, while allocation helpers either reserve arbitrary bytes or allocate all unbacked tensors in bulk with padding/alignment awareness.【F:include/ggml/torch.h†L177-L211】【F:src/ggml-torch.cpp†L962-L1134】 Synchronisation and `graph_compute` forward directly to ggml. Backends can be moved but not copied.

```cpp
Backend cpu = Backend::cpu(/*n_threads=*/4);
BackendBuffer weights = cpu.alloc_tensors(model.parameters());
for (const Tensor & t : model.parameters()) {
    t.assign_buffer(weights);
}
```

### BackendScheduler
Wraps `ggml_backend_sched_t` and exposes creation from one or more backends, graph reservation/allocation, compute (sync or async), tensor/backend association, and eval callback registration.【F:include/ggml/torch.h†L213-L257】【F:src/ggml-torch.cpp†L1136-L1283】 The scheduler caches backend handles and buffer types for later lookups; `release` frees the scheduler.

## Context management
Beyond allocation helpers, `Context::allocate_tensors` can bulk-allocate all tensors in the context either on a specific backend or through a buffer type. Both routes return a `BackendBuffer` that the caller can use to initialise or reset ggml-managed memory.【F:include/ggml/torch.h†L277-L284】【F:src/ggml-torch.cpp†L1344-L1364】 Context lifetime is reference-counted through `shared_ptr`, ensuring tensors retain their creator.

## Module system

### Base module utilities
`nn::Module` tracks registered parameters, buffers, and submodules in ordered `std::map`s and provides recursive enumeration methods similar to PyTorch's `named_parameters` and `buffers`. Registration enforces that tensors and modules share the same ggml context.【F:include/ggml/torch.h†L288-L322】【F:src/ggml-torch.cpp†L1368-L1460】 Subclasses override `forward`.

### Provided layers
* **Linear**: dense affine transform with optional bias; `forward` performs matmul and broadcasted bias add.【F:include/ggml/torch.h†L326-L342】【F:src/ggml-torch.cpp†L1465-L1491】
* **RotaryEmbedding**: stores rope configuration and exposes `apply` helpers for rotating query/key tensors in-place on ggml graph ops.【F:include/ggml/torch.h†L344-L363】【F:src/ggml-torch.cpp†L1493-L1537】
* **FeedForward**: combines up/gate/down projections with SiLU activation and optional gating branch.【F:include/ggml/torch.h†L365-L384】【F:src/ggml-torch.cpp†L1618-L1656】
* **MultiheadAttention**: builds QKV/O linear projections, optional rope application, view/permute reshapes, and Flash Attention for scaled dot-product attention.【F:include/ggml/torch.h†L386-L420】【F:src/ggml-torch.cpp†L1660-L1741】
* **Embedding**, **LayerNorm**, **RMSNorm**, and **Sequential** mirror their PyTorch counterparts using ggml ops and registered parameters/buffers.【F:include/ggml/torch.h†L422-L490】【F:src/ggml-torch.cpp†L1539-L1776】

```cpp
auto net = std::make_shared<Sequential>(ctx);
net->append("proj", std::make_shared<Linear>(ctx, embed_dim, hidden_dim));
net->append("act", std::make_shared<FeedForward>(ctx, hidden_dim, hidden_dim * 4));
Tensor out = net->forward(input);
```

### Model helper
`Model` extends `Module` with GGUF configuration loading, backend-aware weight placement, and a greedy `generate` loop.
* Configuration values from GGUF are parsed into a `std::variant` map covering scalar and vector types.【F:include/ggml/torch.h†L491-L518】【F:src/ggml-torch.cpp†L1792-L2022】
* Weights are allocated per-backend via an optional resolver callback, with fallback CPU allocations, then populated by streaming tensor data from the GGUF file.【F:include/ggml/torch.h†L516-L518】【F:src/ggml-torch.cpp†L1923-L2022】
* Runtime execution builds per-step graphs, prepares backends/schedulers, uploads prompt tokens, executes the graph synchronously, and selects the next token from the final logits row.【F:include/ggml/torch.h†L519-L545】【F:src/ggml-torch.cpp†L2025-L2269】

```cpp
Model model(ctx);
model.load_weights_from_gguf("model.gguf");
std::vector<int> tokens = model.generate({prompt_token_id}, /*n=*/16);
```

These components provide a foundation for building ggml graphs using a PyTorch-like facade while retaining explicit control over ggml contexts, buffers, and device scheduling.
