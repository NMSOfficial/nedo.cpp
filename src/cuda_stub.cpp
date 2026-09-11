#include "nedo/cuda_backend.hpp"
#include <stdexcept>

namespace nedo::cuda_backend {
bool compiled() noexcept { return false; }
bool available() noexcept { return false; }
std::string device_name() { return {}; }
void matvec(const TensorInfo&, std::span<const std::byte>, const float*, float*) {
    throw std::runtime_error("nedo.cpp was built without CUDA support");
}
}
