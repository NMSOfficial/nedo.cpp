#pragma once
#include "nedo/gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nedo::kernels {
float fp16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_fp16(float f) noexcept;
void rms_norm(const float* x, const uint16_t* w, float* y, size_t n, float eps) noexcept;
void rms_norm_tensor(const float* x, const TensorInfo& w, std::span<const std::byte> bytes,
                     float* y, size_t n, float eps);
void add_inplace(float* dst, const float* src, size_t n) noexcept;
void mul_inplace(float* dst, const float* scale, size_t n) noexcept;
void silu_mul(float* gate, const float* up, size_t n) noexcept;
void softmax(float* x, size_t n) noexcept;
void rope(float* q, float* k, uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
          uint64_t position, float theta) noexcept;

// Process-wide inference device for the current alpha backend. "auto" selects
// CUDA when the extension was built with CUDA and a device is available.
void set_device(std::string_view device);
std::string device();
bool cuda_available() noexcept;
std::string cuda_device_name();

void matvec(const TensorInfo& t, std::span<const std::byte> bytes, const float* x, float* y);
void embedding_row(const TensorInfo& t, std::span<const std::byte> bytes, uint32_t row, float* out);
}
