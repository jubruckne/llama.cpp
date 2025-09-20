#include "ggml/torch.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml::torch {

namespace {

[[noreturn]] void throw_bad_context() {
    throw std::runtime_error("ggml::torch tensor operations require tensors to share the same context");
}

std::shared_ptr<Context> assert_shared(Context * ctx) {
    try {
        return ctx->shared_from_this();
    } catch (const std::bad_weak_ptr &) {
        throw std::runtime_error("ggml::torch::Context must be managed by std::shared_ptr");
    }
}

template <typename It>
ggml_tensor * new_tensor_from_range(ggml_context * ctx, ggml_type type, It begin, It end) {
    const auto dims = std::distance(begin, end);
    if (dims <= 0 || dims > GGML_MAX_DIMS) {
        throw std::invalid_argument("ggml::torch::Context::new_tensor expects between 1 and 4 dimensions");
    }

    std::array<int64_t, GGML_MAX_DIMS> ne{};
    int index = 0;
    for (auto it = begin; it != end; ++it, ++index) {
        const auto dim = *it;
        if (dim <= 0) {
            throw std::invalid_argument("ggml::torch::Context::new_tensor dimensions must be positive");
        }
        ne[index] = dim;
    }

    return ggml_new_tensor(ctx, type, static_cast<int>(dims), ne.data());
}

int normalize_dim(int64_t dim, int64_t ndims) {
    if (ndims <= 0) {
        throw std::invalid_argument("tensor does not have any dimensions");
    }
    int64_t result = dim;
    if (result < 0) {
        result += ndims;
    }
    if (result < 0 || result >= ndims) {
        throw std::out_of_range("dimension index is out of range");
    }
    return static_cast<int>(result);
}

std::array<int, GGML_MAX_DIMS> identity_axes() {
    std::array<int, GGML_MAX_DIMS> axes{};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        axes[i] = i;
    }
    return axes;
}

std::array<int64_t, GGML_MAX_DIMS> to_shape_array(const std::vector<int64_t> & shape) {
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) {
        throw std::invalid_argument("reshape expects between 1 and 4 dimensions");
    }
    std::array<int64_t, GGML_MAX_DIMS> result{1, 1, 1, 1};
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            throw std::invalid_argument("tensor dimensions must be positive");
        }
        result[i] = shape[i];
    }
    return result;
}

int tensor_ndims(const ggml_tensor * tensor) {
    int n = ggml_n_dims(tensor);
    return n <= 0 ? 1 : n;
}

template <typename T, typename U>
std::vector<T> convert_numeric_array(const U * data, size_t count) {
    std::vector<T> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(static_cast<T>(data[i]));
    }
    return result;
}

std::vector<bool> convert_bool_array(const int8_t * data, size_t count) {
    std::vector<bool> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(data[i] != 0);
    }
    return result;
}

std::vector<std::string> convert_string_array(const struct gguf_context * ctx, int64_t key_id, size_t count) {
    std::vector<std::string> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(gguf_get_arr_str(ctx, key_id, i));
    }
    return result;
}

} // namespace

Tensor::Tensor(ggml_tensor * tensor, std::shared_ptr<Context> ctx)
    : tensor_(tensor), context_(std::move(ctx)) {
}

Context & Tensor::context() const {
    require_defined();
    return *context_;
}

std::vector<int64_t> Tensor::sizes() const {
    require_defined();
    const int ndims = tensor_ndims(tensor_);
    std::vector<int64_t> result(ndims);
    for (int i = 0; i < ndims; ++i) {
        result[i] = tensor_->ne[i];
    }
    return result;
}

int64_t Tensor::size(int64_t dim) const {
    require_defined();
    const auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("tensor has no dimensions");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    return dims[axis];
}

int64_t Tensor::dim() const {
    require_defined();
    return tensor_ndims(tensor_);
}

int64_t Tensor::numel() const {
    require_defined();
    return ggml_nelements(tensor_);
}

void Tensor::require_defined() const {
    if (!tensor_) {
        throw std::runtime_error("ggml::torch::Tensor is undefined");
    }
    if (!context_) {
        throw std::runtime_error("ggml::torch::Tensor does not have an owning context");
    }
}

void Tensor::require_same_context(const Tensor & other) const {
    if (context_.get() != other.context_.get()) {
        throw_bad_context();
    }
}

void Tensor::require_broadcastable(const Tensor & other) const {
    require_defined();
    other.require_defined();
    if (!ggml_can_repeat(tensor_, other.tensor_)) {
        throw std::invalid_argument("tensor shapes are not broadcast-compatible");
    }
}

Tensor Tensor::wrap(ggml_tensor * tensor) const {
    return Tensor(tensor, context_);
}

Tensor Tensor::permute_internal(const std::array<int, GGML_MAX_DIMS> & axes) const {
    require_defined();
    return wrap(ggml_permute(context_->raw(), tensor_, axes[0], axes[1], axes[2], axes[3]));
}

Tensor Tensor::reshape_internal(const std::array<int64_t, GGML_MAX_DIMS> & shape, int dims) const {
    require_defined();
    auto * ctx = context_->raw();
    switch (dims) {
        case 1:
            return wrap(ggml_reshape_1d(ctx, tensor_, shape[0]));
        case 2:
            return wrap(ggml_reshape_2d(ctx, tensor_, shape[0], shape[1]));
        case 3:
            return wrap(ggml_reshape_3d(ctx, tensor_, shape[0], shape[1], shape[2]));
        case 4:
            return wrap(ggml_reshape_4d(ctx, tensor_, shape[0], shape[1], shape[2], shape[3]));
        default:
            throw std::invalid_argument("reshape expects between 1 and 4 dimensions");
    }
}

Tensor Tensor::view_with_shape(const std::array<int64_t, GGML_MAX_DIMS> & shape, size_t offset) const {
    require_defined();
    auto * ctx = context_->raw();
    const int ndims = tensor_ndims(tensor_);
    switch (ndims) {
        case 1:
            return wrap(ggml_view_1d(ctx, tensor_, shape[0], offset));
        case 2:
            return wrap(ggml_view_2d(ctx, tensor_, shape[0], shape[1], tensor_->nb[1], offset));
        case 3:
            return wrap(ggml_view_3d(ctx, tensor_, shape[0], shape[1], shape[2], tensor_->nb[1], tensor_->nb[2], offset));
        default:
            return wrap(ggml_view_4d(ctx, tensor_, shape[0], shape[1], shape[2], shape[3], tensor_->nb[1], tensor_->nb[2], tensor_->nb[3], offset));
    }
}

Tensor Tensor::reduce_rows_like(int64_t dim, bool keepdim, bool mean) const {
    require_defined();
    const auto dims = sizes();
    if (dims.empty()) {
        return wrap(mean ? ggml_mean(context_->raw(), tensor_) : ggml_sum(context_->raw(), tensor_));
    }

    const int ndims = static_cast<int>(dims.size());
    const int axis  = normalize_dim(dim, ndims);

    if (ndims == 1) {
        ggml_tensor * reduced_raw = mean ? ggml_mean(context_->raw(), tensor_) : ggml_sum(context_->raw(), tensor_);
        Tensor reduced = wrap(reduced_raw);
        if (keepdim) {
            return reduced.reshape({1});
        }
        return reduced;
    }

    Tensor input = *this;
    std::array<int, GGML_MAX_DIMS> perm = identity_axes();
    int perm_len = ndims;

    if (axis != 0 && ndims > 1) {
        std::vector<int> order;
        order.reserve(ndims);
        order.push_back(axis);
        for (int i = 0; i < ndims; ++i) {
            if (i != axis) {
                order.push_back(i);
            }
        }
        for (int i = 0; i < perm_len; ++i) {
            perm[i] = order[i];
        }
        input = input.permute_internal(perm);
    }

    ggml_tensor * reduced_raw = mean ? ggml_mean(context_->raw(), input.tensor_)
                                     : ggml_sum_rows(context_->raw(), input.tensor_);
    Tensor reduced = wrap(reduced_raw);

    if (axis != 0 && ndims > 1) {
        std::array<int, GGML_MAX_DIMS> inv = identity_axes();
        for (int i = 0; i < perm_len; ++i) {
            inv[perm[i]] = i;
        }
        reduced = reduced.permute_internal(inv);
    }

    if (!keepdim) {
        auto out_shape = reduced.sizes();
        if (!out_shape.empty()) {
            out_shape.erase(out_shape.begin() + axis);
            if (out_shape.empty()) {
                return reduced;
            }
            return reduced.reshape(out_shape);
        }
    }

    return reduced;
}

Tensor Tensor::add(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return wrap(ggml_add(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::matmul(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return wrap(ggml_mul_mat(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::mul(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return wrap(ggml_mul(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::sub(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return wrap(ggml_sub(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::div(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return wrap(ggml_div(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::scale(float factor) const {
    require_defined();
    return wrap(ggml_scale(context_->raw(), tensor_, factor));
}

Tensor Tensor::neg() const {
    require_defined();
    return wrap(ggml_neg(context_->raw(), tensor_));
}

Tensor Tensor::sum() const {
    require_defined();
    return wrap(ggml_sum(context_->raw(), tensor_));
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    return reduce_rows_like(dim, keepdim, false);
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    return reduce_rows_like(dim, keepdim, true);
}

Tensor Tensor::softmax() const {
    require_defined();
    return wrap(ggml_soft_max(context_->raw(), tensor_));
}

Tensor Tensor::silu() const {
    require_defined();
    return wrap(ggml_silu(context_->raw(), tensor_));
}

Tensor Tensor::gelu(bool approximate) const {
    require_defined();
    auto * ctx = context_->raw();
    return wrap(approximate ? ggml_gelu_quick(ctx, tensor_) : ggml_gelu(ctx, tensor_));
}

Tensor Tensor::relu() const {
    require_defined();
    return wrap(ggml_relu(context_->raw(), tensor_));
}

Tensor Tensor::layer_norm(float eps) const {
    require_defined();
    return wrap(ggml_norm(context_->raw(), tensor_, eps));
}

Tensor Tensor::rms_norm(float eps) const {
    require_defined();
    return wrap(ggml_rms_norm(context_->raw(), tensor_, eps));
}

Tensor Tensor::diag_mask_inf(int n_past) const {
    require_defined();
    return wrap(ggml_diag_mask_inf(context_->raw(), tensor_, n_past));
}

Tensor Tensor::diag_mask_zero(int n_past) const {
    require_defined();
    return wrap(ggml_diag_mask_zero(context_->raw(), tensor_, n_past));
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    require_defined();
    const int ndims = tensor_ndims(tensor_);
    const int axis0 = normalize_dim(dim0, ndims);
    const int axis1 = normalize_dim(dim1, ndims);
    if (axis0 == axis1) {
        return *this;
    }
    auto axes = identity_axes();
    std::swap(axes[axis0], axes[axis1]);
    return permute_internal(axes);
}

Tensor Tensor::permute(std::initializer_list<int64_t> dims) const {
    require_defined();
    const int ndims = tensor_ndims(tensor_);
    if (dims.size() > GGML_MAX_DIMS) {
        throw std::invalid_argument("permute expects at most 4 dimensions");
    }

    std::array<int, GGML_MAX_DIMS> axes = identity_axes();
    std::array<bool, GGML_MAX_DIMS> seen{};
    int index = 0;
    for (int64_t dim : dims) {
        int axis = normalize_dim(dim, ndims);
        if (seen[axis]) {
            throw std::invalid_argument("permute axes must be unique");
        }
        seen[axis] = true;
        axes[index++] = axis;
    }
    for (int axis = 0; axis < ndims; ++axis) {
        if (!seen[axis]) {
            axes[index++] = axis;
            seen[axis] = true;
        }
    }
    return permute_internal(axes);
}

Tensor Tensor::reshape(std::initializer_list<int64_t> shape) const {
    return reshape(std::vector<int64_t>(shape));
}

Tensor Tensor::reshape(const std::vector<int64_t> & shape) const {
    auto shape_array = to_shape_array(shape);
    return reshape_internal(shape_array, static_cast<int>(shape.size()));
}

Tensor Tensor::view_as(const Tensor & other) const {
    require_defined();
    other.require_defined();
    if (numel() != other.numel()) {
        throw std::invalid_argument("view_as requires tensors with the same number of elements");
    }
    return reshape(other.sizes());
}

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    require_defined();
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot flatten scalar tensor");
    }

    const int ndims = static_cast<int>(dims.size());
    const int start = normalize_dim(start_dim, ndims);
    const int end = normalize_dim(end_dim, ndims);
    if (start > end) {
        throw std::invalid_argument("flatten start dimension must not exceed end dimension");
    }

    int64_t flattened = 1;
    for (int i = start; i <= end; ++i) {
        flattened *= dims[i];
    }

    std::vector<int64_t> target_shape;
    target_shape.reserve(dims.size() - (end - start));
    for (int i = 0; i < start; ++i) {
        target_shape.push_back(dims[i]);
    }
    target_shape.push_back(flattened);
    for (int i = end + 1; i < ndims; ++i) {
        target_shape.push_back(dims[i]);
    }

    if (target_shape.empty()) {
        target_shape.push_back(1);
    }

    return reshape(target_shape);
}

Tensor Tensor::contiguous() const {
    require_defined();
    return wrap(ggml_cont(context_->raw(), tensor_));
}

Tensor Tensor::to(ggml_type type) const {
    require_defined();
    return wrap(ggml_cast(context_->raw(), tensor_, type));
}

Tensor Tensor::clone() const {
    require_defined();
    return wrap(ggml_dup(context_->raw(), tensor_));
}

Tensor Tensor::repeat(std::initializer_list<int64_t> repeats) const {
    require_defined();
    if (repeats.size() == 0 || repeats.size() > GGML_MAX_DIMS) {
        throw std::invalid_argument("repeat expects between 1 and 4 repetitions");
    }

    const auto dims = sizes();
    const size_t target_dims = std::max(static_cast<size_t>(tensor_ndims(tensor_)), repeats.size());
    std::vector<int64_t> target_shape(target_dims, 1);

    size_t index = 0;
    for (int64_t repeat : repeats) {
        if (repeat <= 0) {
            throw std::invalid_argument("repeat factors must be positive");
        }
        const int64_t base = index < dims.size() ? dims[index] : 1;
        target_shape[index] = base * repeat;
        ++index;
    }
    for (; index < target_dims; ++index) {
        target_shape[index] = index < dims.size() ? dims[index] : 1;
    }

    Tensor reference = context_->new_tensor(tensor_->type, target_shape);
    return wrap(ggml_repeat(context_->raw(), tensor_, reference.tensor_));
}

Tensor Tensor::repeat_like(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    require_broadcastable(other);
    return wrap(ggml_repeat(context_->raw(), tensor_, other.tensor_));
}

Tensor Tensor::expand(std::initializer_list<int64_t> shape) const {
    return expand(std::vector<int64_t>(shape));
}

Tensor Tensor::expand(const std::vector<int64_t> & shape) const {
    require_defined();
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) {
        throw std::invalid_argument("expand expects between 1 and 4 dimensions");
    }

    auto dims = sizes();
    const size_t ndims = dims.size();
    const size_t target_dims = shape.size();
    if (target_dims < ndims) {
        throw std::invalid_argument("expand target must have at least as many dimensions as the tensor");
    }

    const size_t offset = target_dims - ndims;
    std::vector<int64_t> target(shape.begin(), shape.end());
    for (size_t i = 0; i < target_dims; ++i) {
        int64_t desired = target[i];
        if (desired == -1) {
            desired = (i < offset) ? 1 : dims[i - offset];
        }
        if (desired <= 0) {
            throw std::invalid_argument("expand dimensions must be positive");
        }

        int64_t source = (i < offset) ? 1 : dims[i - offset];
        if (source != desired && source != 1) {
            throw std::invalid_argument("cannot expand dimension with size greater than 1");
        }
        target[i] = desired;
    }

    Tensor reference = context_->new_tensor(tensor_->type, target);
    return wrap(ggml_repeat(context_->raw(), tensor_, reference.tensor_));
}

Tensor Tensor::expand_as(const Tensor & other) const {
    require_defined();
    other.require_defined();
    require_same_context(other);
    return expand(other.sizes());
}

Tensor Tensor::index_select(const Tensor & indices) const {
    require_defined();
    indices.require_defined();
    require_same_context(indices);
    return wrap(ggml_get_rows(context_->raw(), tensor_, indices.tensor_));
}

Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, float beta, float alpha) const {
    require_defined();
    mat1.require_defined();
    mat2.require_defined();
    require_same_context(mat1);
    require_same_context(mat2);

    auto * ctx = context_->raw();
    ggml_tensor * base = tensor_;
    if (beta != 1.0f) {
        base = ggml_scale(ctx, base, beta);
    }

    ggml_tensor * product = ggml_mul_mat(ctx, mat1.tensor_, mat2.tensor_);
    if (alpha != 1.0f) {
        product = ggml_scale(ctx, product, alpha);
    }

    return wrap(ggml_add(ctx, base, product));
}

void Tensor::assign_buffer(const BackendBuffer & buffer) const {
    require_defined();
    if (!buffer) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    if (tensor_->view_src != nullptr) {
        throw std::invalid_argument("cannot assign backend buffer to tensor view");
    }
    if (tensor_->buffer != nullptr && tensor_->buffer != buffer.raw()) {
        throw std::invalid_argument("tensor is already allocated on a different buffer");
    }
    const auto status = ggml_backend_buffer_init_tensor(buffer.raw(), tensor_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("failed to initialise tensor on backend buffer");
    }
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    require_defined();
    auto dims = sizes();
    const int ndims = static_cast<int>(dims.size());
    const int axis = normalize_dim(dim, ndims + 1);
    dims.insert(dims.begin() + axis, 1);
    return reshape(dims);
}

Tensor Tensor::squeeze(int64_t dim) const {
    require_defined();
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot squeeze scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    if (dims[axis] != 1) {
        throw std::invalid_argument("cannot squeeze dimension with size greater than 1");
    }
    dims.erase(dims.begin() + axis);
    if (dims.empty()) {
        dims.push_back(1);
    }
    return reshape(dims);
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    require_defined();
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot narrow scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    const int64_t axis_size = dims[axis];
    if (start < 0 || start >= axis_size) {
        throw std::out_of_range("narrow start is out of range");
    }
    if (length < 0) {
        length = axis_size - start;
    }
    if (start + length > axis_size) {
        throw std::out_of_range("narrow length exceeds dimension size");
    }

    std::array<int64_t, GGML_MAX_DIMS> shape{1, 1, 1, 1};
    for (size_t i = 0; i < dims.size(); ++i) {
        shape[i] = dims[i];
    }
    shape[axis] = length;
    const size_t offset = static_cast<size_t>(start) * tensor_->nb[axis];
    return view_with_shape(shape, offset);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    require_defined();
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot select from scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    int64_t axis_size = dims[axis];
    int64_t idx = index;
    if (idx < 0) {
        idx += axis_size;
    }
    if (idx < 0 || idx >= axis_size) {
        throw std::out_of_range("select index is out of range");
    }
    return narrow(axis, idx, 1).squeeze(axis);
}

std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    if (chunks <= 0) {
        throw std::invalid_argument("number of chunks must be positive");
    }
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot chunk scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    const int64_t axis_size = dims[axis];
    const int64_t base = axis_size / chunks;
    const int64_t rem  = axis_size % chunks;

    std::vector<Tensor> result;
    result.reserve(static_cast<size_t>(chunks));
    int64_t offset = 0;
    for (int64_t i = 0; i < chunks; ++i) {
        int64_t length = base + (i < rem ? 1 : 0);
        if (length == 0) {
            break;
        }
        result.push_back(narrow(axis, offset, length));
        offset += length;
    }
    return result;
}

std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    if (split_size <= 0) {
        throw std::invalid_argument("split size must be positive");
    }
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot split scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    const int64_t axis_size = dims[axis];

    std::vector<Tensor> result;
    int64_t offset = 0;
    while (offset < axis_size) {
        const int64_t length = std::min(split_size, axis_size - offset);
        result.push_back(narrow(axis, offset, length));
        offset += length;
    }
    return result;
}

std::vector<Tensor> Tensor::split_with_sizes(const std::vector<int64_t> & split_sizes, int64_t dim) const {
    if (split_sizes.empty()) {
        throw std::invalid_argument("split_with_sizes expects at least one split size");
    }
    auto dims = sizes();
    if (dims.empty()) {
        throw std::out_of_range("cannot split scalar tensor");
    }
    const int axis = normalize_dim(dim, static_cast<int64_t>(dims.size()));
    const int64_t axis_size = dims[axis];

    int64_t total = 0;
    for (int64_t size : split_sizes) {
        if (size <= 0) {
            throw std::invalid_argument("split sizes must be positive");
        }
        total += size;
    }
    if (total != axis_size) {
        throw std::invalid_argument("split sizes must sum to the dimension length");
    }

    std::vector<Tensor> result;
    result.reserve(split_sizes.size());
    int64_t offset = 0;
    for (int64_t size : split_sizes) {
        result.push_back(narrow(axis, offset, size));
        offset += size;
    }
    return result;
}

Tensor Tensor::rope(const Tensor & positions, const RopeConfig & config) const {
    return rope(positions, config, Tensor());
}

Tensor Tensor::rope(const Tensor & positions, const RopeConfig & config, Tensor freq_factors) const {
    require_defined();
    positions.require_defined();
    require_same_context(positions);
    if (freq_factors.defined()) {
        require_same_context(freq_factors);
    }
    ggml_tensor * freq = freq_factors.defined() ? freq_factors.tensor_ : nullptr;
    return wrap(ggml_rope_ext(context_->raw(),
                              tensor_,
                              positions.tensor_,
                              freq,
                              config.n_dims,
                              config.mode,
                              config.n_ctx_orig,
                              config.freq_base,
                              config.freq_scale,
                              config.ext_factor,
                              config.attn_factor,
                              config.beta_fast,
                              config.beta_slow));
}

Tensor Tensor::flash_attention(const Tensor & key,
                               const Tensor & value,
                               float scale,
                               float max_bias,
                               float logit_softcap) const {
    return flash_attention(key, value, Tensor(), scale, max_bias, logit_softcap);
}

Tensor Tensor::flash_attention(const Tensor & key,
                               const Tensor & value,
                               Tensor mask,
                               float scale,
                               float max_bias,
                               float logit_softcap) const {
    require_defined();
    key.require_defined();
    value.require_defined();
    require_same_context(key);
    require_same_context(value);
    ggml_tensor * mask_tensor = nullptr;
    if (mask.defined()) {
        require_same_context(mask);
        mask_tensor = mask.tensor_;
    }
    return wrap(ggml_flash_attn_ext(context_->raw(), tensor_, key.tensor_, value.tensor_, mask_tensor, scale, max_bias, logit_softcap));
}

Tensor Tensor::concat(const Tensor & first, const Tensor & second, int64_t dim) {
    if (!first.defined()) {
        return second;
    }
    if (!second.defined()) {
        return first;
    }
    first.require_same_context(second);
    const int ndims = tensor_ndims(first.tensor_);
    const int axis  = normalize_dim(dim, ndims);
    return first.wrap(ggml_concat(first.context().raw(), first.tensor_, second.tensor_, axis));
}

Tensor Tensor::concat(const std::vector<Tensor> & tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat expects at least one tensor");
    }
    Tensor result = tensors.front();
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = concat(result, tensors[i], dim);
    }
    return result;
}

BackendBuffer::BackendBuffer(ggml_backend_buffer_t buffer)
    : buffer_(buffer) {
}

BackendBuffer::~BackendBuffer() {
    reset();
}

BackendBuffer::BackendBuffer(BackendBuffer && other) noexcept
    : buffer_(other.buffer_) {
    other.buffer_ = nullptr;
}

BackendBuffer & BackendBuffer::operator=(BackendBuffer && other) noexcept {
    if (this != &other) {
        reset();
        buffer_       = other.buffer_;
        other.buffer_ = nullptr;
    }
    return *this;
}

void BackendBuffer::reset() {
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
}

void BackendBuffer::reset_allocation() const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    ggml_backend_buffer_reset(buffer_);
}

void BackendBuffer::init_tensor(const Tensor & tensor) const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    auto * raw = tensor.raw();
    if (!raw) {
        throw std::invalid_argument("cannot initialise undefined tensor on backend buffer");
    }
    const auto status = ggml_backend_buffer_init_tensor(buffer_, raw);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("failed to initialise tensor on backend buffer");
    }
}

void BackendBuffer::set_usage(enum ggml_backend_buffer_usage usage) const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    ggml_backend_buffer_set_usage(buffer_, usage);
}

ggml_backend_buffer_type_t BackendBuffer::type() const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    return ggml_backend_buffer_get_type(buffer_);
}

size_t BackendBuffer::size() const {
    if (!buffer_) {
        return 0;
    }
    return ggml_backend_buffer_get_size(buffer_);
}

size_t BackendBuffer::alignment() const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    return ggml_backend_buffer_get_alignment(buffer_);
}

size_t BackendBuffer::max_size() const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    return ggml_backend_buffer_get_max_size(buffer_);
}

size_t BackendBuffer::alloc_size(const Tensor & tensor) const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    auto * raw = tensor.raw();
    if (!raw) {
        throw std::invalid_argument("cannot query allocation size for undefined tensor");
    }
    return ggml_backend_buffer_get_alloc_size(buffer_, raw);
}

bool BackendBuffer::is_host() const {
    if (!buffer_) {
        throw std::invalid_argument("backend buffer must be valid");
    }
    return ggml_backend_buffer_is_host(buffer_);
}

Backend::Backend(ggml_backend_t backend, bool take_ownership)
    : backend_(backend), owns_(take_ownership) {
    if (!backend_) {
        throw std::invalid_argument("backend handle must be non-null");
    }
}

Backend::~Backend() {
    release();
}

Backend::Backend(Backend && other) noexcept
    : backend_(other.backend_), owns_(other.owns_) {
    other.backend_ = nullptr;
    other.owns_    = false;
}

Backend & Backend::operator=(Backend && other) noexcept {
    if (this != &other) {
        release();
        backend_      = other.backend_;
        owns_         = other.owns_;
        other.backend_ = nullptr;
        other.owns_    = false;
    }
    return *this;
}

Backend Backend::cpu(int n_threads) {
    ggml_backend_t handle = ggml_backend_cpu_init();
    if (!handle) {
        throw std::runtime_error("failed to initialise CPU backend");
    }
    if (n_threads > 0) {
        ggml_backend_cpu_set_n_threads(handle, n_threads);
    }
    return Backend(handle, true);
}

Backend Backend::gpu(int device_index, const std::string & params) {
    if (device_index < 0) {
        device_index = 0;
    }
    ggml_backend_dev_t selected = nullptr;
    int current_index = 0;
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            if (current_index == device_index) {
                selected = dev;
                break;
            }
            ++current_index;
        }
    }
    if (!selected) {
        throw std::runtime_error("requested GPU device is not available");
    }
    ggml_backend_t handle = ggml_backend_dev_init(selected, params.empty() ? nullptr : params.c_str());
    if (!handle) {
        throw std::runtime_error("failed to initialise GPU backend");
    }
    return Backend(handle, true);
}

Backend Backend::by_type(enum ggml_backend_dev_type type, const std::string & params) {
    ggml_backend_t handle = ggml_backend_init_by_type(type, params.empty() ? nullptr : params.c_str());
    if (!handle) {
        throw std::runtime_error("failed to initialise backend by type");
    }
    return Backend(handle, true);
}

Backend Backend::by_name(const std::string & name, const std::string & params) {
    ggml_backend_t handle = ggml_backend_init_by_name(name.c_str(), params.empty() ? nullptr : params.c_str());
    if (!handle) {
        throw std::runtime_error("failed to initialise backend by name");
    }
    return Backend(handle, true);
}

ggml_backend_buffer_type_t Backend::default_buffer_type() const {
    if (!backend_) {
        throw std::runtime_error("backend is not initialised");
    }
    return ggml_backend_get_default_buffer_type(backend_);
}

BackendBuffer Backend::alloc_buffer(size_t size) const {
    if (!backend_) {
        throw std::runtime_error("backend is not initialised");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend_, size);
    if (!buffer) {
        throw std::runtime_error("failed to allocate backend buffer");
    }
    return BackendBuffer(buffer);
}

BackendBuffer Backend::alloc_tensors(const std::vector<Tensor> & tensors,
                                     enum ggml_backend_buffer_usage usage) const {
    if (!backend_) {
        throw std::runtime_error("backend is not initialised");
    }
    if (tensors.empty()) {
        return BackendBuffer();
    }
    ggml_backend_buffer_type_t buffer_type = default_buffer_type();
    if (!buffer_type) {
        throw std::runtime_error("backend does not provide a default buffer type");
    }
    const size_t alignment = ggml_backend_get_alignment(backend_);
    size_t required_size   = 0;
    int tensors_to_allocate = 0;
    for (const auto & tensor : tensors) {
        auto * raw = tensor.raw();
        if (!raw) {
            throw std::invalid_argument("cannot allocate undefined tensor");
        }
        if (raw->view_src != nullptr) {
            continue;
        }
        if (raw->buffer != nullptr) {
            continue;
        }
        size_t alloc_size = ggml_backend_buft_get_alloc_size(buffer_type, raw);
        if (alloc_size == 0) {
            continue;
        }
        required_size = GGML_PAD(required_size, alignment);
        required_size += GGML_PAD(alloc_size, alignment);
        ++tensors_to_allocate;
    }
    if (tensors_to_allocate == 0) {
        return BackendBuffer();
    }
    BackendBuffer buffer = alloc_buffer(required_size);
    buffer.reset_allocation();
    for (const auto & tensor : tensors) {
        auto * raw = tensor.raw();
        if (!raw || raw->view_src != nullptr || raw->buffer != nullptr) {
            continue;
        }
        buffer.init_tensor(tensor);
    }
    if (usage != GGML_BACKEND_BUFFER_USAGE_ANY) {
        buffer.set_usage(usage);
    }
    return buffer;
}

void Backend::synchronize() const {
    if (backend_) {
        ggml_backend_synchronize(backend_);
    }
}

ggml_status Backend::graph_compute(struct ggml_cgraph * graph) const {
    if (!backend_) {
        throw std::runtime_error("backend is not initialised");
    }
    return ggml_backend_graph_compute(backend_, graph);
}

void Backend::release() {
    if (backend_ && owns_) {
        ggml_backend_free(backend_);
    }
    backend_ = nullptr;
    owns_    = false;
}

BackendScheduler::BackendScheduler(ggml_backend_sched_t sched,
                                   std::vector<ggml_backend_t> backends,
                                   std::vector<ggml_backend_buffer_type_t> buffer_types)
    : sched_(sched), backends_(std::move(backends)), buffer_types_(std::move(buffer_types)) {
}

BackendScheduler::~BackendScheduler() {
    release();
}

BackendScheduler::BackendScheduler(BackendScheduler && other) noexcept
    : sched_(other.sched_), backends_(std::move(other.backends_)), buffer_types_(std::move(other.buffer_types_)) {
    other.sched_ = nullptr;
}

BackendScheduler & BackendScheduler::operator=(BackendScheduler && other) noexcept {
    if (this != &other) {
        release();
        sched_        = other.sched_;
        backends_     = std::move(other.backends_);
        buffer_types_ = std::move(other.buffer_types_);
        other.sched_  = nullptr;
    }
    return *this;
}

BackendScheduler BackendScheduler::create(const std::vector<Backend> & backends,
                                          const std::vector<ggml_backend_buffer_type_t> & buffer_types,
                                          size_t graph_size,
                                          bool parallel,
                                          bool op_offload) {
    if (backends.empty()) {
        throw std::invalid_argument("backend scheduler requires at least one backend");
    }
    std::vector<ggml_backend_t> handles;
    handles.reserve(backends.size());
    for (const auto & backend : backends) {
        if (!backend.defined()) {
            throw std::invalid_argument("backend scheduler cannot use undefined backend");
        }
        handles.push_back(backend.raw());
    }
    std::vector<ggml_backend_buffer_type_t> buft_local;
    ggml_backend_buffer_type_t * buft_ptr = nullptr;
    if (!buffer_types.empty()) {
        if (buffer_types.size() != backends.size()) {
            throw std::invalid_argument("buffer type list must match number of backends");
        }
        buft_local = buffer_types;
        buft_ptr   = buft_local.data();
    }
    ggml_backend_sched_t sched = ggml_backend_sched_new(handles.data(), buft_ptr, static_cast<int>(handles.size()), graph_size, parallel, op_offload);
    if (!sched) {
        throw std::runtime_error("failed to initialise backend scheduler");
    }
    return BackendScheduler(sched, std::move(handles), std::move(buft_local));
}

ggml_backend_t BackendScheduler::backend_handle(size_t index) const {
    if (index >= backends_.size()) {
        throw std::out_of_range("backend index is out of range");
    }
    return backends_[index];
}

void BackendScheduler::reset() const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    ggml_backend_sched_reset(sched_);
}

void BackendScheduler::set_eval_callback(ggml_backend_sched_eval_callback callback, void * user_data) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    ggml_backend_sched_set_eval_callback(sched_, callback, user_data);
}

bool BackendScheduler::reserve(struct ggml_cgraph * graph) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    return ggml_backend_sched_reserve(sched_, graph);
}

bool BackendScheduler::alloc_graph(struct ggml_cgraph * graph) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    return ggml_backend_sched_alloc_graph(sched_, graph);
}

ggml_status BackendScheduler::graph_compute(struct ggml_cgraph * graph) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    return ggml_backend_sched_graph_compute(sched_, graph);
}

ggml_status BackendScheduler::graph_compute_async(struct ggml_cgraph * graph) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    return ggml_backend_sched_graph_compute_async(sched_, graph);
}

void BackendScheduler::synchronize() const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    ggml_backend_sched_synchronize(sched_);
}

void BackendScheduler::set_tensor_backend(const Tensor & tensor, const Backend & backend) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    auto * raw = tensor.raw();
    if (!raw) {
        throw std::invalid_argument("cannot set backend for undefined tensor");
    }
    if (!backend.defined()) {
        throw std::invalid_argument("cannot set tensor backend to undefined backend");
    }
    if (std::find(backends_.begin(), backends_.end(), backend.raw()) == backends_.end()) {
        throw std::invalid_argument("backend is not managed by this scheduler");
    }
    ggml_backend_sched_set_tensor_backend(sched_, raw, backend.raw());
}

ggml_backend_t BackendScheduler::get_tensor_backend(const Tensor & tensor) const {
    if (!sched_) {
        throw std::runtime_error("backend scheduler is not initialised");
    }
    auto * raw = tensor.raw();
    if (!raw) {
        throw std::invalid_argument("cannot query backend for undefined tensor");
    }
    return ggml_backend_sched_get_tensor_backend(sched_, raw);
}

void BackendScheduler::release() {
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
}

std::shared_ptr<Context> Context::create(const ggml_init_params & params) {
    return std::shared_ptr<Context>(new Context(params));
}

Context::Context(const ggml_init_params & params) {
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("failed to initialise ggml context");
    }
}

Context::~Context() {
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

Tensor Context::wrap(ggml_tensor * tensor) {
    if (!tensor) {
        throw std::invalid_argument("cannot wrap null ggml_tensor");
    }
    return Tensor(tensor, assert_shared(this));
}

Tensor Context::new_tensor(ggml_type type, std::initializer_list<int64_t> shape) {
    auto * tensor = new_tensor_from_range(ctx_, type, shape.begin(), shape.end());
    return wrap(tensor);
}

Tensor Context::new_tensor(ggml_type type, const std::vector<int64_t> & shape) {
    auto * tensor = new_tensor_from_range(ctx_, type, shape.begin(), shape.end());
    return wrap(tensor);
}

Tensor Context::new_f32(float value) {
    auto * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
    *static_cast<float *>(tensor->data) = value;
    return wrap(tensor);
}

Tensor Context::new_i32(std::initializer_list<int32_t> values) {
    if (values.size() == 1) {
        return new_i32(*values.begin());
    }
    auto * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, static_cast<int64_t>(values.size()));
    auto * data   = static_cast<int32_t *>(tensor->data);
    int64_t index = 0;
    for (auto value : values) {
        data[index++] = value;
    }
    return wrap(tensor);
}

Tensor Context::new_i32(int32_t value) {
    auto * tensor = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, 1);
    *static_cast<int32_t *>(tensor->data) = value;
    return wrap(tensor);
}

BackendBuffer Context::allocate_tensors(const Backend & backend) {
    if (!backend.defined()) {
        throw std::invalid_argument("cannot allocate tensors without a valid backend");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_, backend.raw());
    if (!buffer) {
        throw std::runtime_error("failed to allocate tensors on backend");
    }
    return BackendBuffer(buffer);
}

BackendBuffer Context::allocate_tensors(ggml_backend_buffer_type_t buffer_type) {
    if (!buffer_type) {
        throw std::invalid_argument("buffer type must be non-null");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buffer_type);
    if (!buffer) {
        throw std::runtime_error("failed to allocate tensors for buffer type");
    }
    return BackendBuffer(buffer);
}

namespace nn {

Module::Module(std::shared_ptr<Context> ctx)
    : ctx_(std::move(ctx)) {
    if (!ctx_) {
        throw std::invalid_argument("ggml::torch::nn::Module requires a valid context");
    }
}

Tensor & Module::register_parameter(const std::string & name, const Tensor & tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument("cannot register undefined tensor as parameter");
    }
    auto inserted = parameters_.emplace(name, tensor);
    if (!inserted.second) {
        inserted.first->second = tensor;
    }
    return inserted.first->second;
}

Tensor & Module::register_buffer(const std::string & name, const Tensor & tensor) {
    if (!tensor.defined()) {
        throw std::invalid_argument("cannot register undefined tensor as buffer");
    }
    auto inserted = buffers_.emplace(name, tensor);
    if (!inserted.second) {
        inserted.first->second = tensor;
    }
    return inserted.first->second;
}

std::shared_ptr<Module> Module::register_module(const std::string & name, std::shared_ptr<Module> module) {
    if (!module) {
        throw std::invalid_argument("cannot register null module");
    }
    if (module->ctx().get() != ctx_.get()) {
        throw std::invalid_argument("registered module must share the same ggml context");
    }
    modules_[name] = std::move(module);
    return modules_[name];
}

std::vector<Tensor> Module::parameters(bool recurse) const {
    std::vector<Tensor> result;
    result.reserve(parameters_.size());
    for (const auto & kv : parameters_) {
        result.push_back(kv.second);
    }
    if (recurse) {
        for (const auto & kv : modules_) {
            auto child = kv.second;
            if (child) {
                auto child_params = child->parameters(true);
                result.insert(result.end(), child_params.begin(), child_params.end());
            }
        }
    }
    return result;
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters(bool recurse) const {
    std::vector<std::pair<std::string, Tensor>> result;
    for (const auto & kv : parameters_) {
        result.emplace_back(kv.first, kv.second);
    }
    if (recurse) {
        for (const auto & kv : modules_) {
            const auto & prefix = kv.first;
            const auto & module = kv.second;
            if (!module) {
                continue;
            }
            for (const auto & child : module->named_parameters(true)) {
                result.emplace_back(prefix + "." + child.first, child.second);
            }
        }
    }
    return result;
}

std::vector<Tensor> Module::buffers(bool recurse) const {
    std::vector<Tensor> result;
    for (const auto & kv : buffers_) {
        result.push_back(kv.second);
    }
    if (recurse) {
        for (const auto & kv : modules_) {
            const auto & module = kv.second;
            if (module) {
                auto child_buffers = module->buffers(true);
                result.insert(result.end(), child_buffers.begin(), child_buffers.end());
            }
        }
    }
    return result;
}

} // namespace nn

Linear::Linear(std::shared_ptr<Context> context, int64_t in_features, int64_t out_features, bool bias, ggml_type type)
    : Module(std::move(context)), in_features_(in_features), out_features_(out_features) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("ggml::torch::Linear requires positive feature sizes");
    }

    auto shared = ctx();
    weight_ = register_parameter("weight", shared->new_tensor(type, {in_features, out_features}));
    if (bias) {
        bias_ = register_parameter("bias", shared->new_tensor(type, {out_features}));
    }
}

Tensor Linear::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::Linear::forward requires a defined input tensor");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::Linear::forward expects the input tensor to originate from the same context");
    }

    auto output = weight_.matmul(input);
    if (bias_.defined()) {
        output = output.add(bias_.repeat_like(output));
    }
    return output;
}

RotaryEmbedding::RotaryEmbedding(std::shared_ptr<Context> context, int64_t dims, Tensor::RopeConfig config)
    : Module(std::move(context)), dims_(dims), config_(config) {
    if (dims_ <= 0) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding requires a positive dimension");
    }
    if (config_.n_dims == 0) {
        config_.n_dims = static_cast<int>(dims_);
    }
}

Tensor RotaryEmbedding::forward(const Tensor & positions) {
    if (!positions.defined()) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding::forward expects defined position indices");
    }
    if (&positions.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding::forward expects positions from the same context");
    }
    // The rotary embedding does not modify the position tensor directly; users should
    // invoke apply() with query and key states. Returning the positions allows the
    // module to be chained in sequential containers when desired.
    return positions;
}

std::pair<Tensor, Tensor> RotaryEmbedding::apply(const Tensor & query,
                                                 const Tensor & key,
                                                 const Tensor & positions) const {
    return apply(query, key, positions, Tensor());
}

std::pair<Tensor, Tensor> RotaryEmbedding::apply(const Tensor & query,
                                                 const Tensor & key,
                                                 const Tensor & positions,
                                                 Tensor freq_factors) const {
    if (!query.defined() || !key.defined()) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding::apply requires defined query and key tensors");
    }
    if (&query.context() != ctx().get() || &key.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding::apply expects tensors from the same context");
    }
    if (&positions.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::RotaryEmbedding::apply expects positions from the same context");
    }
    Tensor rotated_query = query.rope(positions, config_, freq_factors);
    Tensor rotated_key   = key.rope(positions, config_, freq_factors);
    return {rotated_query, rotated_key};
}

Embedding::Embedding(std::shared_ptr<Context> context, int64_t num_embeddings, int64_t embedding_dim, ggml_type type)
    : Module(std::move(context)), num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    if (num_embeddings <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("ggml::torch::Embedding requires positive dimensions");
    }

    weight_ = register_parameter("weight", this->ctx()->new_tensor(type, {embedding_dim, num_embeddings}));
}

Tensor Embedding::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::Embedding::forward requires defined indices");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::Embedding::forward expects indices allocated from the same context");
    }

    return weight_.index_select(input);
}

LayerNorm::LayerNorm(std::shared_ptr<Context> context,
                     std::vector<int64_t> normalized_shape,
                     float eps,
                     bool elementwise_affine,
                     ggml_type type)
    : Module(std::move(context)),
      normalized_shape_(std::move(normalized_shape)),
      eps_(eps),
      elementwise_affine_(elementwise_affine) {
    if (normalized_shape_.empty()) {
        throw std::invalid_argument("ggml::torch::LayerNorm expects a non-empty normalized_shape");
    }

    if (elementwise_affine_) {
        weight_ = register_parameter("weight", this->ctx()->new_tensor(type, normalized_shape_));
        bias_   = register_parameter("bias",   this->ctx()->new_tensor(type, normalized_shape_));
    }
}

Tensor LayerNorm::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::LayerNorm::forward requires a defined input tensor");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::LayerNorm::forward expects the input tensor to originate from the same context");
    }

    auto result = input.layer_norm(eps_);
    if (elementwise_affine_ && weight_.defined()) {
        result = result.mul(weight_.repeat_like(result));
        if (bias_.defined()) {
            result = result.add(bias_.repeat_like(result));
        }
    }
    return result;
}

RMSNorm::RMSNorm(std::shared_ptr<Context> context, int64_t normalized_shape, float eps, ggml_type type)
    : Module(std::move(context)), eps_(eps) {
    if (normalized_shape <= 0) {
        throw std::invalid_argument("ggml::torch::RMSNorm requires a positive normalized_shape");
    }

    weight_ = register_parameter("weight", this->ctx()->new_tensor(type, {normalized_shape}));
}

Tensor RMSNorm::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::RMSNorm::forward requires a defined input tensor");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::RMSNorm::forward expects the input tensor to originate from the same context");
    }

    auto result = input.rms_norm(eps_);
    return result.mul(weight_.repeat_like(result));
}

FeedForward::FeedForward(std::shared_ptr<Context> context,
                         int64_t embed_dim,
                         int64_t hidden_dim,
                         bool gated,
                         ggml_type type)
    : Module(std::move(context)), gated_(gated) {
    if (embed_dim <= 0 || hidden_dim <= 0) {
        throw std::invalid_argument("ggml::torch::FeedForward requires positive dimensions");
    }

    auto shared = ctx();
    up_proj_   = std::make_shared<Linear>(shared, embed_dim, hidden_dim, true, type);
    down_proj_ = std::make_shared<Linear>(shared, hidden_dim, embed_dim, true, type);
    register_module("up_proj", up_proj_);
    register_module("down_proj", down_proj_);

    if (gated_) {
        gate_proj_ = std::make_shared<Linear>(shared, embed_dim, hidden_dim, true, type);
        register_module("gate_proj", gate_proj_);
    }
}

Tensor FeedForward::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::FeedForward::forward expects defined inputs");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::FeedForward::forward expects inputs from the same context");
    }

    Tensor hidden = up_proj_->forward(input);
    if (gated_ && gate_proj_) {
        Tensor gate = gate_proj_->forward(input).silu();
        hidden = hidden.mul(gate);
    } else {
        hidden = hidden.silu();
    }
    return down_proj_->forward(hidden);
}

MultiheadAttention::MultiheadAttention(std::shared_ptr<Context> context,
                                       int64_t embed_dim,
                                       int64_t num_heads,
                                       bool bias,
                                       ggml_type type,
                                       Tensor::RopeConfig rope)
    : Module(std::move(context)),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      rope_(rope) {
    if (embed_dim_ <= 0 || num_heads_ <= 0) {
        throw std::invalid_argument("ggml::torch::MultiheadAttention requires positive dimensions");
    }
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument("embed dimension must be divisible by the number of heads");
    }

    head_dim_ = embed_dim_ / num_heads_;
    if (rope_.n_dims == 0) {
        rope_.n_dims = static_cast<int>(head_dim_);
    }

    auto shared = ctx();
    q_proj_ = std::make_shared<Linear>(shared, embed_dim_, embed_dim_, bias, type);
    k_proj_ = std::make_shared<Linear>(shared, embed_dim_, embed_dim_, bias, type);
    v_proj_ = std::make_shared<Linear>(shared, embed_dim_, embed_dim_, bias, type);
    o_proj_ = std::make_shared<Linear>(shared, embed_dim_, embed_dim_, bias, type);

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("o_proj", o_proj_);
}

Tensor MultiheadAttention::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::MultiheadAttention::forward expects defined input");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::MultiheadAttention::forward expects input from the same context");
    }
    if (attention_mask_.defined() && &attention_mask_.context() != ctx().get()) {
        throw std::invalid_argument("attention mask must originate from the same context");
    }
    if (position_ids_.defined() && &position_ids_.context() != ctx().get()) {
        throw std::invalid_argument("position ids must originate from the same context");
    }
    if (freq_factors_.defined() && &freq_factors_.context() != ctx().get()) {
        throw std::invalid_argument("frequency factors must originate from the same context");
    }

    Tensor query = q_proj_->forward(input);
    Tensor key   = k_proj_->forward(input);
    Tensor value = v_proj_->forward(input);

    if (rope_.n_dims > 0 && position_ids_.defined()) {
        query = query.rope(position_ids_, rope_, freq_factors_);
        key   = key.rope(position_ids_, rope_, freq_factors_);
    }

    auto dims = query.sizes();
    if (dims.size() < 2) {
        throw std::invalid_argument("multihead attention expects inputs with sequence dimension");
    }
    const int64_t seq_len = dims[1];
    const int64_t batch   = dims.size() > 2 ? dims[2] : 1;

    std::vector<int64_t> qkv_shape = {head_dim_, num_heads_, seq_len, batch};
    query = query.view(qkv_shape).permute({0, 2, 1, 3});
    key   = key.view(qkv_shape).permute({0, 2, 1, 3});
    value = value.view(qkv_shape).permute({0, 2, 1, 3});

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    Tensor attn;
    if (attention_mask_.defined()) {
        attn = query.flash_attention(key, value, attention_mask_, scale, 0.0f, 0.0f);
    } else {
        attn = query.flash_attention(key, value, scale, 0.0f, 0.0f);
    }
    attn = attn.permute({0, 3, 1, 2}).contiguous();

    Tensor merged = attn.view({head_dim_ * num_heads_, seq_len, batch});
    Tensor output = o_proj_->forward(merged);
    return output;
}

Sequential::Sequential(std::shared_ptr<Context> context)
    : Module(std::move(context)) {
}

Sequential::Sequential(std::shared_ptr<Context> context, std::initializer_list<std::shared_ptr<Module>> modules)
    : Sequential(std::move(context)) {
    size_t index = 0;
    for (auto module : modules) {
        append(std::to_string(index++), std::move(module));
    }
}

Tensor Sequential::forward(const Tensor & input) {
    if (!input.defined()) {
        throw std::invalid_argument("ggml::torch::Sequential::forward requires a defined input tensor");
    }
    if (&input.context() != ctx().get()) {
        throw std::invalid_argument("ggml::torch::Sequential::forward expects the input tensor to originate from the same context");
    }

    Tensor current = input;
    for (const auto & entry : ordered_modules_) {
        if (entry.second) {
            current = entry.second->forward(current);
        }
    }
    return current;
}

Sequential & Sequential::append(const std::string & name, std::shared_ptr<Module> module) {
    ordered_modules_.emplace_back(name, register_module(name, std::move(module)));
    return *this;
}

Sequential & Sequential::append(std::shared_ptr<Module> module) {
    const auto name = std::to_string(ordered_modules_.size());
    return append(name, std::move(module));
}

Model::Model(std::shared_ptr<Context> context)
    : nn::Module(std::move(context)) {
}

void Model::clear_config() {
    config_.clear();
    parameter_buffers_.clear();
}

void Model::load_weights_from_gguf(const std::string & path, BackendResolver resolver) {
    struct gguf_init_params params {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };

    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!ctx) {
        throw std::runtime_error("failed to open GGUF file for weight loading");
    }

    config_.clear();
    parameter_buffers_.clear();

    const int64_t n_kv = gguf_get_n_kv(ctx.get());
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf_get_key(ctx.get(), i);
        switch (gguf_get_kv_type(ctx.get(), i)) {
            case GGUF_TYPE_UINT8:
                config_[key] = static_cast<uint64_t>(gguf_get_val_u8(ctx.get(), i));
                break;
            case GGUF_TYPE_INT8:
                config_[key] = static_cast<int64_t>(gguf_get_val_i8(ctx.get(), i));
                break;
            case GGUF_TYPE_UINT16:
                config_[key] = static_cast<uint64_t>(gguf_get_val_u16(ctx.get(), i));
                break;
            case GGUF_TYPE_INT16:
                config_[key] = static_cast<int64_t>(gguf_get_val_i16(ctx.get(), i));
                break;
            case GGUF_TYPE_UINT32:
                config_[key] = static_cast<uint64_t>(gguf_get_val_u32(ctx.get(), i));
                break;
            case GGUF_TYPE_INT32:
                config_[key] = static_cast<int64_t>(gguf_get_val_i32(ctx.get(), i));
                break;
            case GGUF_TYPE_FLOAT32:
                config_[key] = static_cast<double>(gguf_get_val_f32(ctx.get(), i));
                break;
            case GGUF_TYPE_BOOL:
                config_[key] = gguf_get_val_bool(ctx.get(), i);
                break;
            case GGUF_TYPE_STRING:
                config_[key] = std::string(gguf_get_val_str(ctx.get(), i));
                break;
            case GGUF_TYPE_UINT64:
                config_[key] = gguf_get_val_u64(ctx.get(), i);
                break;
            case GGUF_TYPE_INT64:
                config_[key] = gguf_get_val_i64(ctx.get(), i);
                break;
            case GGUF_TYPE_FLOAT64:
                config_[key] = gguf_get_val_f64(ctx.get(), i);
                break;
            case GGUF_TYPE_ARRAY: {
                const size_t count = gguf_get_arr_n(ctx.get(), i);
                const gguf_type elem_type = gguf_get_arr_type(ctx.get(), i);
                switch (elem_type) {
                    case GGUF_TYPE_UINT8: {
                        const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = std::vector<uint8_t>(data, data + count);
                    } break;
                    case GGUF_TYPE_INT8: {
                        const auto * data = static_cast<const int8_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_numeric_array<int64_t>(data, count);
                    } break;
                    case GGUF_TYPE_UINT16: {
                        const auto * data = static_cast<const uint16_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_numeric_array<uint64_t>(data, count);
                    } break;
                    case GGUF_TYPE_INT16: {
                        const auto * data = static_cast<const int16_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_numeric_array<int64_t>(data, count);
                    } break;
                    case GGUF_TYPE_UINT32: {
                        const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_numeric_array<uint64_t>(data, count);
                    } break;
                    case GGUF_TYPE_INT32: {
                        const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_numeric_array<int64_t>(data, count);
                    } break;
                    case GGUF_TYPE_UINT64: {
                        const auto * data = static_cast<const uint64_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = std::vector<uint64_t>(data, data + count);
                    } break;
                    case GGUF_TYPE_INT64: {
                        const auto * data = static_cast<const int64_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = std::vector<int64_t>(data, data + count);
                    } break;
                    case GGUF_TYPE_FLOAT32: {
                        const auto * data = static_cast<const float *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = std::vector<float>(data, data + count);
                    } break;
                    case GGUF_TYPE_FLOAT64: {
                        const auto * data = static_cast<const double *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = std::vector<double>(data, data + count);
                    } break;
                    case GGUF_TYPE_BOOL: {
                        const auto * data = static_cast<const int8_t *>(gguf_get_arr_data(ctx.get(), i));
                        config_[key] = convert_bool_array(data, count);
                    } break;
                    case GGUF_TYPE_STRING:
                        config_[key] = convert_string_array(ctx.get(), i, count);
                        break;
                    default:
                        throw std::runtime_error("unsupported GGUF array type in configuration");
                }
            } break;
            default:
                throw std::runtime_error("unsupported GGUF value type in configuration");
        }
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open GGUF file for reading tensor data");
    }

    const auto parameters = named_parameters(true);

    Backend fallback_backend;
    const Backend * fallback_ptr = nullptr;
    if (!resolver) {
        fallback_backend = Backend::cpu();
        if (!fallback_backend.defined()) {
            throw std::runtime_error("failed to create CPU backend for weight allocation");
        }
        fallback_ptr = &fallback_backend;
    }

    std::unordered_map<const Backend *, std::vector<Tensor>> allocations;
    allocations.reserve(parameters.size());

    for (const auto & entry : parameters) {
        const auto & name = entry.first;
        const auto & tensor = entry.second;
        if (!tensor.defined()) {
            throw std::runtime_error("encountered undefined parameter while loading weights");
        }

        ggml_tensor * raw = tensor.raw();
        if (raw == nullptr) {
            throw std::runtime_error("encountered parameter without underlying ggml tensor");
        }
        if (raw->view_src != nullptr) {
            continue;
        }

        const Backend * backend_ptr = nullptr;
        if (resolver) {
            backend_ptr = resolver(name, tensor);
        }
        if (!backend_ptr) {
            backend_ptr = fallback_ptr;
        }

        if (raw->buffer == nullptr) {
            if (!backend_ptr) {
                throw std::runtime_error("no backend available to allocate parameter buffer");
            }
            allocations[backend_ptr].push_back(tensor);
        }
    }

    for (auto & alloc : allocations) {
        const Backend * backend = alloc.first;
        if (!backend || !backend->defined()) {
            throw std::runtime_error("attempted to allocate parameters on an undefined backend");
        }
        BackendBuffer buffer = backend->alloc_tensors(alloc.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buffer.set_usage(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & tensor : alloc.second) {
            tensor.assign_buffer(buffer);
        }
        parameter_buffers_.push_back(std::move(buffer));
    }

    const size_t data_offset = gguf_get_data_offset(ctx.get());
    std::vector<char> buffer;

    for (const auto & entry : parameters) {
        const auto & name = entry.first;
        const auto & tensor = entry.second;
        if (!tensor.defined()) {
            throw std::runtime_error("encountered undefined parameter while uploading weights");
        }
        ggml_tensor * raw = tensor.raw();
        if (raw->view_src != nullptr) {
            continue;
        }
        if (raw->buffer == nullptr) {
            throw std::runtime_error("parameter '" + name + "' does not have an allocated backend buffer");
        }

        const int64_t tensor_index = gguf_find_tensor(ctx.get(), name.c_str());
        if (tensor_index < 0) {
            throw std::runtime_error("tensor '" + name + "' not found in GGUF file");
        }

        const size_t tensor_size = gguf_get_tensor_size(ctx.get(), tensor_index);
        const size_t expected_size = ggml_nbytes(raw);
        if (tensor_size != expected_size) {
            throw std::runtime_error("size mismatch while loading tensor '" + name + "'");
        }

        const enum ggml_type tensor_type = gguf_get_tensor_type(ctx.get(), tensor_index);
        if (tensor_type != raw->type) {
            throw std::runtime_error("type mismatch while loading tensor '" + name + "'");
        }

        const size_t tensor_offset = data_offset + gguf_get_tensor_offset(ctx.get(), tensor_index);

        buffer.resize(tensor_size);
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(tensor_offset), std::ios::beg);
        if (!stream.good()) {
            throw std::runtime_error("failed to seek to tensor data for '" + name + "'");
        }
        if (tensor_size > 0) {
            stream.read(buffer.data(), static_cast<std::streamsize>(tensor_size));
            if (stream.gcount() != static_cast<std::streamsize>(tensor_size)) {
                throw std::runtime_error("failed to read tensor data for '" + name + "'");
            }
        }

        ggml_backend_buffer_set_usage(raw->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (tensor_size > 0) {
            ggml_backend_tensor_set(raw, buffer.data(), 0, tensor_size);
        }
    }
}

Model::ExecutionBackends Model::prepare_execution_backends() const {
    ExecutionBackends execution_backends;

    Backend cpu_backend = Backend::cpu();
    if (!cpu_backend.defined()) {
        throw std::runtime_error("failed to initialise CPU backend for generation");
    }

    ggml_backend_buffer_type_t cpu_buffer_type = cpu_backend.default_buffer_type();
    if (!cpu_buffer_type) {
        throw std::runtime_error("CPU backend does not provide a default buffer type");
    }

    execution_backends.buffer_to_index[cpu_buffer_type] = execution_backends.backends.size();
    execution_backends.buffer_types.push_back(cpu_buffer_type);
    execution_backends.backends.push_back(std::move(cpu_backend));
    execution_backends.cpu_index = 0;

    for (const auto & buffer : parameter_buffers_) {
        ggml_backend_buffer_type_t type = buffer.type();
        if (!type) {
            continue;
        }

        if (execution_backends.buffer_to_index.find(type) != execution_backends.buffer_to_index.end()) {
            continue;
        }

        ggml_backend_dev_t device = ggml_backend_buft_get_device(type);
        if (device == nullptr) {
            execution_backends.buffer_to_index[type] = execution_backends.cpu_index;
            continue;
        }

        ggml_backend_t handle = ggml_backend_dev_init(device, nullptr);
        if (!handle) {
            throw std::runtime_error("failed to initialise backend for parameter buffer");
        }

        Backend backend(handle, true);
        execution_backends.buffer_to_index[type] = execution_backends.backends.size();
        execution_backends.buffer_types.push_back(type);
        execution_backends.backends.push_back(std::move(backend));
    }

    if (execution_backends.empty()) {
        throw std::runtime_error("no execution backend available for generation");
    }

    return execution_backends;
}

void Model::collect_tensor_placements(const ExecutionBackends & execution_backends,
                                      std::vector<std::pair<Tensor, size_t>> & placements) const {
    placements.clear();

    const auto params  = named_parameters(true);
    const auto buffers = this->buffers(true);
    placements.reserve(params.size() + buffers.size());

    auto record_tensor = [&](const Tensor & tensor) {
        if (!tensor.defined()) {
            return;
        }
        auto * raw = tensor.raw();
        if (!raw || raw->view_src != nullptr || raw->buffer == nullptr) {
            return;
        }
        ggml_backend_buffer_type_t type = ggml_backend_buffer_get_type(raw->buffer);
        auto it = execution_backends.buffer_to_index.find(type);
        if (it == execution_backends.buffer_to_index.end()) {
            return;
        }
        placements.emplace_back(tensor, it->second);
    };

    for (const auto & entry : params) {
        record_tensor(entry.second);
    }
    for (const auto & tensor : buffers) {
        record_tensor(tensor);
    }
}

BackendScheduler Model::create_scheduler(const ExecutionBackends & execution_backends,
                                         size_t graph_nodes) const {
    if (execution_backends.empty()) {
        throw std::runtime_error("no execution backend available for generation");
    }

    const size_t graph_size = std::max<size_t>(graph_nodes + 16, size_t{16});
    return BackendScheduler::create(execution_backends.backends,
                                    execution_backends.buffer_types,
                                    graph_size,
                                    false,
                                    false);
}

void Model::assign_backends(BackendScheduler & scheduler,
                            const ExecutionBackends & execution_backends,
                            const std::vector<std::pair<Tensor, size_t>> & placements,
                            const Tensor & input_tokens) const {
    for (const auto & placement : placements) {
        const size_t backend_index = placement.second;
        if (backend_index >= execution_backends.backends.size()) {
            throw std::runtime_error("tensor placement references an unknown backend");
        }
        scheduler.set_tensor_backend(placement.first, execution_backends.backends[backend_index]);
    }

    if (execution_backends.cpu_index >= execution_backends.backends.size()) {
        throw std::runtime_error("CPU backend index is out of range");
    }

    scheduler.set_tensor_backend(input_tokens, execution_backends.backends[execution_backends.cpu_index]);
}

void Model::upload_prompt_tokens(const Tensor & input_tokens, const std::vector<int> & tokens) const {
    if (!input_tokens.defined()) {
        throw std::invalid_argument("input tensor for prompt upload must be defined");
    }

    if (input_tokens.raw()->type != GGML_TYPE_I32) {
        throw std::invalid_argument("prompt tensor must use 32-bit integer storage");
    }

    std::vector<int32_t> prompt_data(tokens.begin(), tokens.end());
    if (!prompt_data.empty()) {
        ggml_backend_tensor_set(input_tokens.raw(), prompt_data.data(), 0, prompt_data.size() * sizeof(int32_t));
    }
}

int Model::select_next_token(const Tensor & logits) const {
    if (logits.raw()->type != GGML_TYPE_F32) {
        throw std::invalid_argument("logits tensor must be in float32 format for token selection");
    }

    const auto shape = logits.sizes();
    if (shape.empty()) {
        throw std::runtime_error("model forward pass returned scalar logits");
    }

    const int64_t vocab_size = shape.back();
    if (vocab_size <= 0) {
        throw std::runtime_error("model forward pass produced invalid logits shape");
    }

    const size_t total_values = static_cast<size_t>(logits.numel());
    if (total_values % static_cast<size_t>(vocab_size) != 0) {
        throw std::runtime_error("logit tensor size is not divisible by vocabulary size");
    }

    std::vector<float> logit_values(total_values);
    if (!logit_values.empty()) {
        ggml_backend_tensor_get(logits.raw(), logit_values.data(), 0, logit_values.size() * sizeof(float));
    }

    const int64_t rows = static_cast<int64_t>(total_values / static_cast<size_t>(vocab_size));
    if (rows <= 0) {
        throw std::runtime_error("logit tensor does not contain any rows");
    }

    const size_t offset = static_cast<size_t>((rows - 1) * vocab_size);
    auto begin = logit_values.begin() + offset;
    auto end   = begin + vocab_size;
    auto max_it = std::max_element(begin, end);
    if (max_it == end) {
        throw std::runtime_error("failed to select next token from logits");
    }

    return static_cast<int>(std::distance(begin, max_it));
}

std::vector<int> Model::generate(std::vector<int> prompt, int n) {
    if (n < 0) {
        throw std::invalid_argument("number of tokens to generate must be non-negative");
    }

    auto shared_ctx = ctx();
    if (!shared_ctx) {
        throw std::runtime_error("model context is not initialised");
    }

    std::vector<int> tokens = std::move(prompt);
    if (n == 0) {
        return tokens;
    }
    if (tokens.empty()) {
        throw std::invalid_argument("prompt must contain at least one token");
    }

    ExecutionBackends execution_backends = prepare_execution_backends();

    std::vector<std::pair<Tensor, size_t>> placements;

    tokens.reserve(tokens.size() + static_cast<size_t>(n));

    for (int generated = 0; generated < n; ++generated) {
        Tensor input_tokens = shared_ctx->new_tensor(GGML_TYPE_I32, {static_cast<int64_t>(tokens.size())});
        ggml_set_input(input_tokens.raw());

        Tensor logits = forward(input_tokens);
        if (!logits.defined()) {
            throw std::runtime_error("model forward pass produced an undefined tensor");
        }
        if (&logits.context() != shared_ctx.get()) {
            throw std::runtime_error("model forward pass returned a tensor from a different context");
        }
        if (logits.raw()->type != GGML_TYPE_F32) {
            logits = logits.to(GGML_TYPE_F32);
        }
        ggml_set_output(logits.raw());

        struct ggml_cgraph * graph = ggml_new_graph_custom(shared_ctx->raw(), GGML_DEFAULT_GRAPH_SIZE, false);
        if (!graph) {
            throw std::runtime_error("failed to allocate computation graph");
        }

        ggml_build_forward_expand(graph, logits.raw());

        BackendScheduler scheduler = create_scheduler(execution_backends, ggml_graph_n_nodes(graph));

        collect_tensor_placements(execution_backends, placements);
        assign_backends(scheduler, execution_backends, placements, input_tokens);

        if (!scheduler.reserve(graph)) {
            throw std::runtime_error("failed to reserve backend resources for graph");
        }
        if (!scheduler.alloc_graph(graph)) {
            throw std::runtime_error("failed to allocate graph buffers on backends");
        }

        upload_prompt_tokens(input_tokens, tokens);

        const ggml_status status = scheduler.graph_compute(graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph execution failed during generation");
        }
        scheduler.synchronize();

        const int next_token = select_next_token(logits);
        tokens.push_back(next_token);
    }

    return tokens;
}

} // namespace ggml::torch
