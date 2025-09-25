#ifndef GGML_TORCH_H
#define GGML_TORCH_H

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ggml::torch {

class Context;
class Backend;
class BackendBuffer;
class Config;

class BackendScheduler;
class Model;

class Shape {
public:
    Shape();

    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(const std::vector<int64_t> & dims);
    explicit Shape(std::span<const int64_t> dims);

    static Shape scalar();
    static Shape vector(int64_t n);
    static Shape matrix(int64_t rows, int64_t cols);

    template <size_t N>
    explicit Shape(const std::array<int64_t, N> & dims)
        : Shape(dims.begin(), dims.end()) {
    }

    Shape(const int64_t * data, size_t dims);

    const int64_t * data() const { return storage_ref().data(); }

    size_t size() const noexcept { return GGML_MAX_DIMS; }
    size_t dims() const noexcept { return static_cast<size_t>(decode_dims(index_)); }

    const int64_t * begin() const noexcept { return data(); }
    const int64_t * end() const noexcept { return data() + size(); }
    const int64_t * cbegin() const noexcept { return begin(); }
    const int64_t * cend() const noexcept { return end(); }

    int64_t size(size_t dim) const;
    int64_t numel() const;

    int64_t operator[](size_t index) const;
    bool operator==(const Shape & other) const noexcept;
    bool operator!=(const Shape & other) const noexcept { return !(*this == other); }

    operator std::vector<int64_t>() const;
    operator std::array<int64_t, GGML_MAX_DIMS>() const;
    operator const int64_t *() const noexcept { return data(); }
    operator std::span<const int64_t>() const noexcept { return {data(), dims()}; }

    bool is_scalar() const noexcept { return dims() == 0; }
    bool is_vector() const noexcept { return dims() == 1; }
    bool is_matrix() const noexcept { return dims() == 2; }

    int64_t flatten(std::span<const int64_t> indices) const;
    int64_t flatten(std::initializer_list<int64_t> indices) const {
        return flatten(std::span<const int64_t>(indices.begin(), indices.size()));
    }

    std::vector<int64_t> unravel(int64_t index) const;
    void unravel(int64_t index, std::span<int64_t> indices) const;

    template <typename It>
    Shape(It begin, It end);

private:
    static constexpr uint16_t kDimsShift = 13;
    static constexpr uint16_t kIndexMask = (1u << kDimsShift) - 1;
    static constexpr size_t   kMaxShapes = static_cast<size_t>(kIndexMask) + 1;

    static_assert(GGML_MAX_DIMS <= (1u << (16 - kDimsShift)), "ggml::torch::Shape encoding does not support GGML_MAX_DIMS");

    static uint16_t store(const std::array<int64_t, GGML_MAX_DIMS> & storage);
    static const std::array<int64_t, GGML_MAX_DIMS> & storage_at(uint16_t index);

    static constexpr uint16_t encode(uint16_t storage_index, uint8_t dims) noexcept {
        return static_cast<uint16_t>((static_cast<uint16_t>(dims) << kDimsShift) |
                                     (storage_index & kIndexMask));
    }

    static constexpr uint16_t decode_storage_index(uint16_t encoded) noexcept {
        return static_cast<uint16_t>(encoded & kIndexMask);
    }

    static constexpr uint8_t decode_dims(uint16_t encoded) noexcept {
        return static_cast<uint8_t>(encoded >> kDimsShift);
    }

    uint16_t storage_index() const noexcept { return decode_storage_index(index_); }
    const std::array<int64_t, GGML_MAX_DIMS> & storage_ref() const { return storage_at(storage_index()); }

    uint16_t index_ = 0;
};

template <typename It>
Shape::Shape(It begin, It end) {
    std::array<int64_t, GGML_MAX_DIMS> storage{};
    storage.fill(1);

    uint8_t dims = 0;
    for (auto it = begin; it != end; ++it) {
        if (dims >= GGML_MAX_DIMS) {
            throw std::invalid_argument("shape expects at most 4 dimensions");
        }

        const int64_t value = static_cast<int64_t>(*it);
        if (value <= 0) {
            throw std::invalid_argument("shape dimensions must be positive");
        }

        storage[dims++] = value;
    }

    if (dims == 0) {
        index_ = encode(0, 0);
        return;
    }

    index_ = encode(store(storage), dims);
}

std::string qualify_name(std::string_view base, std::string_view name);

template <typename T>
concept KeyPartConcept =
    std::is_convertible_v<std::decay_t<T>, std::string_view> ||
    std::integral<std::remove_reference_t<T>>;

class Tensor {
public:
    Tensor() = default;

    ggml_tensor * raw() const noexcept { return tensor_; }
    bool defined() const noexcept { return tensor_ != nullptr; }

    Context & context() const;

    std::vector<int64_t> sizes() const;
    int64_t size(int64_t dim) const;
    int64_t dim() const;
    int64_t numel() const;

    Tensor add(const Tensor & other) const;
    Tensor matmul(const Tensor & other) const;
    Tensor mm(const Tensor & other) const { return matmul(other); }
    Tensor mul(const Tensor & other) const;
    Tensor sub(const Tensor & other) const;
    Tensor div(const Tensor & other) const;
    Tensor scale(float factor) const;
    Tensor neg() const;
    Tensor sum() const;
    Tensor sum(int64_t dim, bool keepdim = false) const;
    Tensor mean(int64_t dim, bool keepdim = false) const;
    Tensor softmax() const;
    Tensor softmax(int64_t dim) const;
    Tensor silu() const;
    Tensor gelu(bool approximate = true) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor elu() const;
    Tensor leaky_relu(float negative_slope = 0.01f) const;
    Tensor layer_norm(float eps) const;
    Tensor rms_norm(float eps) const;
    Tensor diag_mask_inf(int n_past) const;
    Tensor diag_mask_zero(int n_past) const;

    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(std::initializer_list<int64_t> dims) const;
    Tensor reshape(std::initializer_list<int64_t> shape) const;
    Tensor reshape(const std::vector<int64_t> & shape) const;
    Tensor view(std::initializer_list<int64_t> shape) const { return reshape(shape); }
    Tensor view(const std::vector<int64_t> & shape) const { return reshape(shape); }
    Tensor view_as(const Tensor & other) const;
    Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;
    Tensor contiguous() const;
    Tensor to(ggml_type type) const;
    Tensor clone() const;

    Tensor unsqueeze(int64_t dim) const;
    Tensor squeeze(int64_t dim) const;
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor select(int64_t dim, int64_t index) const;
    std::vector<Tensor> chunk(int64_t chunks, int64_t dim) const;
    std::vector<Tensor> split(int64_t split_size, int64_t dim) const;
    std::vector<Tensor> split_with_sizes(const std::vector<int64_t> & split_sizes, int64_t dim) const;

    Tensor repeat(std::initializer_list<int64_t> repeats) const;

    Tensor repeat_like(const Tensor & other) const;
    Tensor expand(std::initializer_list<int64_t> shape) const;
    Tensor expand(const std::vector<int64_t> & shape) const;
    Tensor expand_as(const Tensor & other) const;
    Tensor index_select(const Tensor & indices) const;
    Tensor addmm(const Tensor & mat1, const Tensor & mat2, float beta = 1.0f, float alpha = 1.0f) const;

    void assign_buffer(const BackendBuffer & buffer) const;

    struct RopeConfig {
        int   n_dims      = 0;
        int   mode        = 0;
        int   n_ctx_orig  = 0;
        float freq_base   = 10000.0f;
        float freq_scale  = 1.0f;
        float ext_factor  = 1.0f;
        float attn_factor = 1.0f;
        float beta_fast   = 32.0f;
        float beta_slow   = 1.0f;
    };

    Tensor rope(const Tensor & positions,
                const RopeConfig & config) const;
    Tensor rope(const Tensor & positions,
                const RopeConfig & config,
                Tensor freq_factors) const;

    Tensor flash_attention(const Tensor & key,
                           const Tensor & value,
                           float scale = 1.0f,
                           float max_bias = 0.0f,
                           float logit_softcap = 0.0f) const;
    Tensor flash_attention(const Tensor & key,
                           const Tensor & value,
                           Tensor mask,
                           float scale,
                           float max_bias,
                           float logit_softcap) const;

    static Tensor concat(const Tensor & first, const Tensor & second, int64_t dim);
    static Tensor concat(const std::vector<Tensor> & tensors, int64_t dim);

private:
    friend class Context;
    friend class Backend;

    Tensor(ggml_tensor * tensor, std::shared_ptr<Context> ctx);

    Tensor wrap(ggml_tensor * tensor) const;
    void require_defined() const;
    void require_same_context(const Tensor & other) const;
    void require_broadcastable(const Tensor & other) const;

    Tensor permute_internal(const std::array<int, GGML_MAX_DIMS> & axes) const;
    Tensor reshape_internal(const Shape & shape) const;
    Tensor view_with_shape(const Shape & shape, size_t offset) const;
    Tensor reduce_rows_like(int64_t dim, bool keepdim, bool mean) const;

private:
    ggml_tensor * tensor_ = nullptr;
    std::shared_ptr<Context> context_;
};

class BackendBuffer {
public:
    BackendBuffer() = default;
    explicit BackendBuffer(ggml_backend_buffer_t buffer);
    ~BackendBuffer();

    BackendBuffer(const BackendBuffer &) = delete;
    BackendBuffer & operator=(const BackendBuffer &) = delete;

    BackendBuffer(BackendBuffer && other) noexcept;
    BackendBuffer & operator=(BackendBuffer && other) noexcept;

    ggml_backend_buffer_t raw() const noexcept { return buffer_; }
    explicit operator bool() const noexcept { return buffer_ != nullptr; }

    void reset();
    void reset_allocation() const;
    void init_tensor(const Tensor & tensor) const;
    void set_usage(enum ggml_backend_buffer_usage usage) const;
    ggml_backend_buffer_type_t type() const;
    size_t size() const;
    size_t alignment() const;
    size_t max_size() const;
    size_t alloc_size(const Tensor & tensor) const;
    bool is_host() const;

private:
    ggml_backend_buffer_t buffer_ = nullptr;
};

class Backend {
public:
    Backend() = default;
    explicit Backend(ggml_backend_t backend, bool take_ownership = true);
    ~Backend();

    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;

    Backend(Backend && other) noexcept;
    Backend & operator=(Backend && other) noexcept;

    static Backend cpu(int n_threads = 0);
    static Backend gpu(int device_index = 0, const std::string & params = {});
    static Backend by_type(enum ggml_backend_dev_type type, const std::string & params = {});
    static Backend by_name(const std::string & name, const std::string & params = {});

    ggml_backend_t raw() const noexcept { return backend_; }
    ggml_backend_buffer_type_t default_buffer_type() const;
    BackendBuffer alloc_buffer(size_t size) const;
    BackendBuffer alloc_tensors(const std::vector<Tensor> & tensors,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_ANY) const;

    void synchronize() const;
    ggml_status graph_compute(struct ggml_cgraph * graph) const;

    bool defined() const noexcept { return backend_ != nullptr; }

private:
    void release();

private:
    ggml_backend_t backend_ = nullptr;
    bool owns_ = false;
};

class BackendScheduler {
public:
    BackendScheduler() = default;
    explicit BackendScheduler(ggml_backend_sched_t sched,
                              std::vector<ggml_backend_t> backends = {},
                              std::vector<ggml_backend_buffer_type_t> buffer_types = {});
    ~BackendScheduler();

    BackendScheduler(const BackendScheduler &) = delete;
    BackendScheduler & operator=(const BackendScheduler &) = delete;

    BackendScheduler(BackendScheduler && other) noexcept;
    BackendScheduler & operator=(BackendScheduler && other) noexcept;

    static BackendScheduler create(const std::vector<Backend> & backends,
                                   const std::vector<ggml_backend_buffer_type_t> & buffer_types = {},
                                   size_t graph_size = GGML_DEFAULT_GRAPH_SIZE,
                                   bool parallel = false,
                                   bool op_offload = true);

    ggml_backend_sched_t raw() const noexcept { return sched_; }
    bool defined() const noexcept { return sched_ != nullptr; }

    size_t num_backends() const noexcept { return backends_.size(); }
    ggml_backend_t backend_handle(size_t index) const;

    void reset() const;
    void set_eval_callback(ggml_backend_sched_eval_callback callback, void * user_data) const;
    bool reserve(struct ggml_cgraph * graph) const;
    bool alloc_graph(struct ggml_cgraph * graph) const;
    ggml_status graph_compute(struct ggml_cgraph * graph) const;
    ggml_status graph_compute_async(struct ggml_cgraph * graph) const;
    void synchronize() const;

    void set_tensor_backend(const Tensor & tensor, const Backend & backend) const;
    ggml_backend_t get_tensor_backend(const Tensor & tensor) const;

private:
    void release();

private:
    ggml_backend_sched_t sched_ = nullptr;
    std::vector<ggml_backend_t> backends_{};
    std::vector<ggml_backend_buffer_type_t> buffer_types_{};
};

class Context : public std::enable_shared_from_this<Context> {
public:
    static std::shared_ptr<Context> create(const ggml_init_params & params);

    ~Context();

    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;

    ggml_context * raw() const noexcept { return ctx_; }

    Tensor wrap(ggml_tensor * tensor) const;
    Tensor new_tensor(ggml_type type, std::initializer_list<int64_t> shape) const;
    Tensor new_tensor(ggml_type type, const std::vector<int64_t> & shape) const;
    Tensor new_f32(float value) const;
    Tensor new_i32(std::initializer_list<int32_t> values) const;
    Tensor new_i32(int32_t value) const;

    BackendBuffer allocate_tensors(const Backend & backend) const;
    BackendBuffer allocate_tensors(ggml_backend_buffer_type_t buffer_type) const;

private:
    explicit Context(const ggml_init_params & params);

    ggml_context * ctx_ = nullptr;
};

namespace nn {

class Module : public std::enable_shared_from_this<Module> {
public:
    explicit Module(const Model * model, std::string name);
    virtual ~Module() = default;

    Module(const Module &) = delete;
    Module & operator=(const Module &) = delete;
    Module(Module &&) noexcept = default;
    Module & operator=(Module &&) noexcept = default;

    virtual Tensor forward(const Tensor & input) = 0;

    Tensor & register_parameter(const std::string & name,
                                std::optional<Shape> shape = {},
                                std::optional<ggml_type> type = {});
    Tensor & register_buffer(const std::string & name, Shape shape, ggml_type type);

    template <typename TModule, typename... Args>
    std::shared_ptr<TModule> register_module(const std::string & name, Args &&... args);

protected:
    const std::map<std::string, std::shared_ptr<Module>> & modules() const { return modules_; }

    Model       & model();
    const Model & model() const;
    const std::string & name() const noexcept { return name_; }

private:
    friend class ggml::torch::Model;

    const Model * model_ = nullptr;
    std::map<std::string, std::shared_ptr<Module>> modules_;
    std::string name_;
};

template <typename TModule, typename... Args>
std::shared_ptr<TModule> Module::register_module(const std::string & name, Args &&... args) {
    if (!model_) {
        throw std::runtime_error("cannot register submodule without an owning model");
    }
    if (name.empty()) {
        throw std::invalid_argument("module name must be non-empty");
    }

    const std::string qualified = qualify_name(name_, name);
    std::shared_ptr<TModule> module;

    if constexpr (std::is_constructible_v<TModule, Model &, std::string, Args...>) {
        module = std::make_shared<TModule>(model(), qualified, std::forward<Args>(args)...);
    } else if constexpr (std::is_constructible_v<TModule, Model *, std::string, Args...>) {
        module = std::make_shared<TModule>(&model(), qualified, std::forward<Args>(args)...);
    } else {
        static_assert(std::is_constructible_v<TModule, Model &, std::string, Args...> ||
                          std::is_constructible_v<TModule, Model *, std::string, Args...>,
                      "register_module requires TModule to be constructible with Model& or Model* and a name");
    }

    auto insertion = modules_.emplace(name, module);
    if (!insertion.second) {
        throw std::invalid_argument("module '" + qualified + "' is already registered");
    }
    return module;
}

} // namespace nn

class Linear : public nn::Module {
public:
    Linear(Model & model,
           int64_t in_features,
           int64_t out_features,
           bool bias = true,
           ggml_type type = GGML_TYPE_F32);
    Linear(Model & model,
           std::string name,
           int64_t in_features,
           int64_t out_features,
           bool bias = true,
           ggml_type type = GGML_TYPE_F32);

    Tensor forward(const Tensor & input) override;

    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }

private:
    Tensor weight_;
    Tensor bias_;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
};

class RotaryEmbedding : public nn::Module {
public:
    RotaryEmbedding(Model & model,
                    int64_t dims,
                    Tensor::RopeConfig rope_config = {});
    RotaryEmbedding(Model & model,
                    std::string name,
                    int64_t dims,
                    Tensor::RopeConfig rope_config = {});

    Tensor forward(const Tensor & positions) override;

    std::pair<Tensor, Tensor> apply(const Tensor & query,
                                    const Tensor & key,
                                    const Tensor & positions) const;
    std::pair<Tensor, Tensor> apply(const Tensor & query,
                                    const Tensor & key,
                                    const Tensor & positions,
                                    Tensor freq_factors) const;

    const Tensor::RopeConfig & config() const { return config_; }

private:
    int64_t dims_ = 0;
    Tensor::RopeConfig config_{};
};

class FeedForward : public nn::Module {
public:
    FeedForward(Model & model,
                int64_t embed_dim,
                int64_t hidden_dim,
                bool gated = true,
                ggml_type type = GGML_TYPE_F32);
    FeedForward(Model & model,
                std::string name,
                int64_t embed_dim,
                int64_t hidden_dim,
                bool gated = true,
                ggml_type type = GGML_TYPE_F32);

    Tensor forward(const Tensor & input) override;

    std::shared_ptr<Linear> gate() const { return gate_proj_; }
    std::shared_ptr<Linear> up() const { return up_proj_; }
    std::shared_ptr<Linear> down() const { return down_proj_; }

private:
    std::shared_ptr<Linear> gate_proj_;
    std::shared_ptr<Linear> up_proj_;
    std::shared_ptr<Linear> down_proj_;
    bool gated_ = true;
};

class MultiheadAttention : public nn::Module {
public:
    MultiheadAttention(Model & model,
                       int64_t embed_dim,
                       int64_t num_heads,
                       bool bias = true,
                       ggml_type type = GGML_TYPE_F32,
                       Tensor::RopeConfig rope = {});
    MultiheadAttention(Model & model,
                       std::string name,
                       int64_t embed_dim,
                       int64_t num_heads,
                       bool bias = true,
                       ggml_type type = GGML_TYPE_F32,
                       Tensor::RopeConfig rope = {});

    Tensor forward(const Tensor & input) override;

    void set_attention_mask(const Tensor & mask) { attention_mask_ = mask; }
    void set_positions(const Tensor & positions) { position_ids_ = positions; }
    void set_frequency_factors(const Tensor & factors) { freq_factors_ = factors; }

    std::shared_ptr<Linear> q_proj() const { return q_proj_; }
    std::shared_ptr<Linear> k_proj() const { return k_proj_; }
    std::shared_ptr<Linear> v_proj() const { return v_proj_; }
    std::shared_ptr<Linear> o_proj() const { return o_proj_; }

private:
    int64_t embed_dim_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    Tensor::RopeConfig rope_{};

    std::shared_ptr<Linear> q_proj_;
    std::shared_ptr<Linear> k_proj_;
    std::shared_ptr<Linear> v_proj_;
    std::shared_ptr<Linear> o_proj_;

    Tensor attention_mask_;
    Tensor position_ids_;
    Tensor freq_factors_;
};

class Embedding : public nn::Module {
public:
    Embedding(Model & model,
              int64_t num_embeddings,
              int64_t embedding_dim,
              ggml_type type = GGML_TYPE_F32);
    Embedding(Model & model,
              std::string name,
              int64_t num_embeddings,
              int64_t embedding_dim,
              ggml_type type = GGML_TYPE_F32);

    Tensor forward(const Tensor & input) override;

    Tensor weight() const { return weight_; }
    int64_t num_embeddings() const { return num_embeddings_; }
    int64_t embedding_dim() const { return embedding_dim_; }

private:
    Tensor weight_;
    int64_t num_embeddings_ = 0;
    int64_t embedding_dim_ = 0;
};

class LayerNorm : public nn::Module {
public:
    LayerNorm(Model & model,
              std::vector<int64_t> normalized_shape,
              float eps = 1e-5f,
              bool elementwise_affine = true,
              ggml_type type = GGML_TYPE_F32);
    LayerNorm(Model & model,
              std::string name,
              std::vector<int64_t> normalized_shape,
              float eps = 1e-5f,
              bool elementwise_affine = true,
              ggml_type type = GGML_TYPE_F32);

    Tensor forward(const Tensor & input) override;

    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    float eps() const { return eps_; }

private:
    std::vector<int64_t> normalized_shape_;
    Tensor weight_;
    Tensor bias_;
    float eps_ = 0.0f;
    bool elementwise_affine_ = true;
};

class RMSNorm : public nn::Module {
public:
    RMSNorm(Model & model,
            int64_t normalized_shape,
            float eps = 1e-5f,
            ggml_type type = GGML_TYPE_F32);
    RMSNorm(Model & model,
            std::string name,
            int64_t normalized_shape,
            float eps = 1e-5f,
            ggml_type type = GGML_TYPE_F32);

    Tensor forward(const Tensor & input) override;

    Tensor weight() const { return weight_; }
    float eps() const { return eps_; }

private:
    Tensor weight_;
    float eps_ = 0.0f;
};

class ReLU : public nn::Module {
public:
    using Module::Module;

    explicit ReLU(Model & model, std::string name = "")
        : Module(&model, std::move(name)) {}

    Tensor forward(const Tensor & input) override;
};

class SiLU : public nn::Module {
public:
    using Module::Module;

    explicit SiLU(Model & model, std::string name = "")
        : Module(&model, std::move(name)) {}

    Tensor forward(const Tensor & input) override;
};

class GELU : public nn::Module {
public:
    GELU(Model & model, std::string name = "", bool approximate = true);

    Tensor forward(const Tensor & input) override;

    bool approximate() const { return approximate_; }

private:
    bool approximate_ = true;
};

class Sigmoid : public nn::Module {
public:
    using Module::Module;

    explicit Sigmoid(Model & model, std::string name = "")
        : Module(&model, std::move(name)) {}

    Tensor forward(const Tensor & input) override;
};

class Tanh : public nn::Module {
public:
    using Module::Module;

    explicit Tanh(Model & model, std::string name = "")
        : Module(&model, std::move(name)) {}

    Tensor forward(const Tensor & input) override;
};

class ELU : public nn::Module {
public:
    ELU(Model & model, std::string name = "", float alpha = 1.0f);

    Tensor forward(const Tensor & input) override;

    float alpha() const { return alpha_; }

private:
    float alpha_ = 1.0f;
};

class LeakyReLU : public nn::Module {
public:
    LeakyReLU(Model & model, std::string name = "", float negative_slope = 0.01f);

    Tensor forward(const Tensor & input) override;

    float negative_slope() const { return negative_slope_; }

private:
    float negative_slope_ = 0.01f;
};

class Softmax : public nn::Module {
public:
    Softmax(Model & model, std::string name = "", int64_t dim = -1);

    Tensor forward(const Tensor & input) override;

    int64_t dim() const { return dim_; }

private:
    int64_t dim_ = -1;
};

class Sequential : public nn::Module {
public:
    explicit Sequential(Model & model, std::string name = "");

    Tensor forward(const Tensor & input) override;

    template <typename TModule, typename... Args>
    Sequential & append(const std::string & name, Args &&... args);

private:
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> ordered_modules_;
};

template <typename TModule, typename... Args>
Sequential & Sequential::append(const std::string & name, Args &&... args) {
    auto module = register_module<TModule>(name, std::forward<Args>(args)...);
    ordered_modules_.emplace_back(name, std::move(module));
    return *this;
}

struct TensorInfo {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    Shape shape{1, 1, 1, 1};
};

struct Value {
    using Variant = std::variant<
        int8_t,
        int16_t,
        int32_t,
        int64_t,
        uint8_t,
        uint16_t,
        uint32_t,
        uint64_t,
        float,
        double,
        bool,
        std::string,
        std::vector<int8_t>,
        std::vector<int16_t>,
        std::vector<int32_t>,
        std::vector<int64_t>,
        std::vector<uint8_t>,
        std::vector<uint16_t>,
        std::vector<uint32_t>,
        std::vector<uint64_t>,
        std::vector<float>,
        std::vector<double>,
        std::vector<bool>,
        std::vector<std::string>,
        TensorInfo>;

    Value() = delete;

    template <typename Key,
              typename T,
              typename DecayedKey   = std::decay_t<Key>,
              typename DecayedValue = std::decay_t<T>,
              typename = std::enable_if_t<!std::is_same_v<DecayedValue, Value> &&
                                          std::is_constructible_v<std::string, DecayedKey>>>
    Value(Key && key, T && value)
        : value_(std::forward<T>(value)), key_(std::forward<Key>(key)) {
    }

    Value(const Value &) = default;
    Value(Value &&) noexcept = default;
    Value & operator=(const Value &) = delete;
    Value & operator=(Value &&) noexcept = delete;

    template <typename T>
    bool is() const {
        return std::holds_alternative<T>(value_);
    }

    template <typename T>
    const T & get() const {
        return std::get<T>(value_);
    }

    std::string_view key() const { return key_; }

    const Variant & variant() const { return value_; }

    // equality by key only
    bool operator==(const Value & other) const noexcept {
        return key_ == other.key_;
    }

    // transparent equality for heterogeneous lookup
    friend bool operator==(const Value & v, std::string_view k) noexcept {
        return v.key_ == k;
    }

    friend bool operator==(std::string_view k, const Value & v) noexcept {
        return k == v.key_;
    }

    struct Hash {
        using is_transparent = void;

        std::size_t operator()(const Value & v) const noexcept {
            return std::hash<std::string_view>{}(v.key_);
        }

        std::size_t operator()(std::string_view k) const noexcept {
            return std::hash<std::string_view>{}(k);
        }
    };

private:
    const Variant value_;
    const std::string key_;
};

class Config {
public:
    using Container = std::unordered_set<Value, Value::Hash, std::equal_to<>>;

    Config() = default;
    explicit Config(std::vector<Value> values);

    Value operator[](std::string key) const;
    size_t size() const noexcept;

    template <typename... Keys>
    bool contains(Keys &&... keys) const {
        static_assert(sizeof...(Keys) > 0, "contains requires at least one key part");
        const std::string key = build_key(std::forward<Keys>(keys)...);
        return find_value(key) != nullptr;
    }

    template <typename T, typename... Keys>
    bool is(Keys &&... keys) const {
        static_assert(sizeof...(Keys) > 0, "is requires at least one key part");
        const std::string key = build_key(std::forward<Keys>(keys)...);
        const Value * value   = find_value(key);
        if (!value) {
            return false;
        }
        return value->template is<T>();
    }

    template <typename T, typename... Keys>
    const T & get(Keys &&... keys) const {
        static_assert(sizeof...(Keys) > 0, "get requires at least one key part");
        const std::string key = build_key(std::forward<Keys>(keys)...);
        return at(key).template get<T>();
    }

    const Container & values() const noexcept { return values_; }

private:
    template <KeyPartConcept... Keys>
    static std::string build_key(Keys &&... keys) {
        std::array<std::string, sizeof...(Keys)> parts{
            key_part_to_string(std::forward<Keys>(keys))...
        };

        size_t total_size = parts.empty() ? 0 : parts.size() - 1; // account for dots
        for (const auto & part : parts) {
            total_size += part.size();
        }

        std::string result;
        result.reserve(total_size);

        for (size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) {
                result.push_back('.');
            }
            result.append(parts[i]);
        }

        return result;
    }

    template <typename Key>
    static std::string key_part_to_string(Key && key) {
        using Decayed = std::decay_t<Key>;
        if constexpr (std::is_convertible_v<Decayed, std::string_view>) {
            return std::string(std::string_view(key));
        } else {
            static_assert(std::is_integral_v<Decayed>,
                          "Config key parts must be string-like or integral");
            if constexpr (std::is_signed_v<Decayed>) {
                return std::to_string(static_cast<long long>(key));
            } else {
                return std::to_string(static_cast<unsigned long long>(key));
            }
        }
    }

    const Value & at(std::string_view key) const;
    const Value * find_value(std::string_view key) const noexcept;

    Container values_;
};

class Model {
public:
    explicit Model(Config config);
    virtual ~Model() = default;

    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    Model(Model &&) noexcept = default;
    Model & operator=(Model &&) noexcept = default;

    virtual Tensor forward(const Tensor & input) = 0;

    const Config * config() const;
    const Context * ctx() const;
    std::shared_ptr<Context> shared_context() const;

    const std::map<std::string, Tensor> & parameters() const;
    const std::map<std::string, Tensor> & buffers() const;

    Tensor & register_parameter(const std::string & name,
                                std::optional<Shape> shape = {},
                                std::optional<ggml_type> type = {});
    Tensor & register_buffer(const std::string & name, Shape shape, ggml_type type);

protected:
    virtual nn::Module & module() = 0;
    virtual const nn::Module & module() const = 0;

private:
    friend class nn::Module;

    using TensorMap = std::map<std::string, Tensor>;

    static std::vector<int64_t> shape_to_dims(const Shape & shape);

    std::shared_ptr<Context> context_;
    Config config_{};
    TensorMap parameters_;
    TensorMap buffers_;
};

class Loader;

class Generator {
public:
    using BackendResolver = std::function<Backend &(const std::string &, const Tensor &)>;

    Generator(const Generator &) = delete;
    Generator & operator=(const Generator &) = delete;

    Generator(Generator &&) noexcept = default;
    Generator & operator=(Generator &&) noexcept = default;

    const Config & config() const { return *model_->config(); }
    Model       & model() { return *model_; }
    const Model & model() const { return *model_; }

    std::vector<int> generate(std::vector<int> prompt, int n);

private:
    Generator(std::shared_ptr<Model> model, std::vector<BackendBuffer> parameter_buffers);

    friend class Loader;

    struct GenerationWorkspace {
        GenerationWorkspace() = default;
        GenerationWorkspace(const GenerationWorkspace &) = delete;
        GenerationWorkspace & operator=(const GenerationWorkspace &) = delete;
        GenerationWorkspace(GenerationWorkspace &&) = default;
        GenerationWorkspace & operator=(GenerationWorkspace &&) = default;

        void clear() {
            *this = GenerationWorkspace{};
        }

        bool empty() const noexcept { return backends.empty(); }

        std::vector<Backend> backends;
        std::vector<ggml_backend_buffer_type_t> buffer_types;
        std::unordered_map<ggml_backend_buffer_type_t, size_t> buffer_to_index;
        size_t cpu_index = 0;
        BackendScheduler scheduler;
        std::vector<std::pair<Tensor, size_t>> cached_placements;
        size_t reserved_graph_nodes = 0;
        size_t max_graph_nodes = 0;
    };

    void prepare_execution_backends(GenerationWorkspace & workspace) const;
    void collect_tensor_placements(const GenerationWorkspace & workspace,
                                   std::vector<std::pair<Tensor, size_t>> & placements) const;
    BackendScheduler create_scheduler(const GenerationWorkspace & workspace,
                                      size_t graph_nodes) const;
    void assign_backends(BackendScheduler & scheduler,
                         const GenerationWorkspace & workspace,
                         const std::vector<std::pair<Tensor, size_t>> & placements,
                         const Tensor & input_tokens) const;
    void upload_prompt_tokens(const Tensor & input_tokens, const std::vector<int> & tokens) const;
    int select_next_token(const Tensor & logits) const;
    GenerationWorkspace & ensure_generation_workspace() const;
    void invalidate_generation_workspace();

    std::shared_ptr<Model> model_;
    std::vector<BackendBuffer> parameter_buffers_;
    mutable GenerationWorkspace generation_workspace_;
    mutable bool generation_workspace_ready_ = false;
};

class Loader {
public:
    using BackendResolver = Generator::BackendResolver;

    template <typename TModel>
    static std::pair<std::shared_ptr<TModel>, Generator>
    load_from_gguf(const std::string & gguf_path, Backend & backend) {
        BackendResolver resolver = [&backend](const std::string &, const Tensor &) -> Backend & {
            return backend;
        };
        return load_from_gguf_with_resolver<TModel>(gguf_path, std::move(resolver));
    }

    template <typename TModel>
    static std::pair<std::shared_ptr<TModel>, Generator>
    load_from_gguf(const std::string & gguf_path, BackendResolver resolver) {
        return load_from_gguf_with_resolver<TModel>(gguf_path, std::move(resolver));
    }

    static Config load_config_from_gguf(const std::string & gguf_path);

private:
    static std::vector<BackendBuffer>
    load_weights_from_gguf(Model & model, const std::string & gguf_path, BackendResolver & resolver);

    template <typename TModel>
    static std::pair<std::shared_ptr<TModel>, Generator>
    load_from_gguf_with_resolver(const std::string & gguf_path, BackendResolver resolver) {
        static_assert(std::is_base_of_v<Model, TModel>,
                      "Loader::load_from_gguf requires TModel to derive from ggml::torch::Model");

        if (!resolver) {
            throw std::invalid_argument("Loader::load_from_gguf requires a backend resolver");
        }

        auto config  = load_config_from_gguf(gguf_path);
        auto model   = std::make_shared<TModel>(std::move(config));

        std::shared_ptr<Model> base_model = model;
        auto parameter_buffers = load_weights_from_gguf(*base_model, gguf_path, resolver);
        Generator generator(base_model, std::move(parameter_buffers));
        return {std::move(model), std::move(generator)};
    }
};

} // namespace ggml::torch

#endif // GGML_TORCH_H
