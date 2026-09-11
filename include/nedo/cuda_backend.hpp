#pragma once
#include "nedo/gguf.hpp"
#include <cstddef>
#include <span>
#include <string>

namespace nedo::cuda_backend {
bool compiled() noexcept;
bool available() noexcept;
std::string device_name();
void matvec(const TensorInfo& t, std::span<const std::byte> bytes, const float* x, float* y);
}
