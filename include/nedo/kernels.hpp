#pragma once
#include "nedo/gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nedo::kernels {
float fp16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_fp16(float f) noexcept;
void rms_norm(const float* x, const uint16_t* w, float* y, size_t n, float eps) noexcept;
void add_inplace(float* dst, const float* src, size_t n) noexcept;
void mul_inplace(float* dst, const float* scale, size_t n) noexcept;
void silu_mul(float* gate, const float* up, size_t n) noexcept;
void softmax(float* x, size_t n) noexcept;
void rope(float* q, float* k, uint32_t n_q_heads, uint32_t n_kv_heads, uint32_t head_dim,
          uint64_t position, float theta) noexcept;
void matvec(const TensorInfo& t, std::span<const std::byte> bytes, const float* x, float* y);
void embedding_row(const TensorInfo& t, std::span<const std::byte> bytes, uint32_t row, float* out);
}
