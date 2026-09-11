#include "nedo/cuda_backend.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nedo::cuda_backend {
namespace {

[[noreturn]] void fail(const char* what, cudaError_t err) {
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(err));
}
void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) fail(what, err);
}

struct DeviceWeight {
    void* ptr{};
    size_t bytes{};
};

struct WeightCache {
    std::mutex mutex;
    std::unordered_map<const std::byte*, DeviceWeight> weights;
    ~WeightCache() {
        for (auto& [_, entry] : weights) if (entry.ptr) cudaFree(entry.ptr);
    }
};

WeightCache& cache() {
    static WeightCache instance;
    return instance;
}

const void* device_weight(std::span<const std::byte> bytes) {
    auto& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    auto it = c.weights.find(bytes.data());
    if (it != c.weights.end()) {
        if (it->second.bytes != bytes.size()) throw std::runtime_error("CUDA weight cache size mismatch");
        return it->second.ptr;
    }
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, bytes.size()), "cudaMalloc(weight)");
    try {
        check(cudaMemcpy(ptr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "upload weight");
    } catch (...) {
        cudaFree(ptr);
        throw;
    }
    c.weights.emplace(bytes.data(), DeviceWeight{ptr, bytes.size()});
    return ptr;
}

struct Scratch {
    float* x{};
    float* y{};
    size_t x_cap{};
    size_t y_cap{};
    ~Scratch() {
        if (x) cudaFree(x);
        if (y) cudaFree(y);
    }
    void reserve(size_t nx, size_t ny) {
        if (nx > x_cap) {
            if (x) check(cudaFree(x), "cudaFree(x scratch)");
            check(cudaMalloc(&x, nx * sizeof(float)), "cudaMalloc(x scratch)");
            x_cap = nx;
        }
        if (ny > y_cap) {
            if (y) check(cudaFree(y), "cudaFree(y scratch)");
            check(cudaMalloc(&y, ny * sizeof(float)), "cudaMalloc(y scratch)");
            y_cap = ny;
        }
    }
};

thread_local Scratch scratch;

__device__ __forceinline__ float block_reduce(float v, float* shared) {
    const unsigned tid = threadIdx.x;
    shared[tid] = v;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    return shared[0];
}

__global__ void matvec_f16_kernel(const __half* w, const float* x, float* y, uint64_t cols) {
    extern __shared__ float shared[];
    const uint64_t row = blockIdx.x;
    const __half* rw = w + row * cols;
    float sum = 0.0f;
    for (uint64_t c = threadIdx.x; c < cols; c += blockDim.x) sum += __half2float(rw[c]) * x[c];
    const float total = block_reduce(sum, shared);
    if (threadIdx.x == 0) y[row] = total;
}

__global__ void matvec_q8_kernel(const unsigned char* w, const float* x, float* y, uint64_t cols) {
    extern __shared__ float shared[];
    const uint64_t row = blockIdx.x;
    const uint64_t row_bytes = (cols / 32) * 34;
    const unsigned char* rw = w + row * row_bytes;
    float sum = 0.0f;
    for (uint64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint64_t blk = c >> 5;
        const uint64_t lane = c & 31;
        const unsigned char* p = rw + blk * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(p));
        const int8_t q = reinterpret_cast<const int8_t*>(p + 2)[lane];
        sum += d * static_cast<float>(q) * x[c];
    }
    const float total = block_reduce(sum, shared);
    if (threadIdx.x == 0) y[row] = total;
}

__global__ void matvec_q4_kernel(const unsigned char* w, const float* x, float* y, uint64_t cols) {
    extern __shared__ float shared[];
    const uint64_t row = blockIdx.x;
    const uint64_t row_bytes = (cols / 32) * 18;
    const unsigned char* rw = w + row * row_bytes;
    float sum = 0.0f;
    for (uint64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint64_t blk = c >> 5;
        const uint64_t lane = c & 31;
        const unsigned char* p = rw + blk * 18;
        const float d = __half2float(*reinterpret_cast<const __half*>(p));
        const unsigned char packed = p[2 + (lane & 15)];
        const int q = lane < 16 ? int(packed & 0x0f) - 8 : int(packed >> 4) - 8;
        sum += d * static_cast<float>(q) * x[c];
    }
    const float total = block_reduce(sum, shared);
    if (threadIdx.x == 0) y[row] = total;
}

} // namespace

bool compiled() noexcept { return true; }

bool available() noexcept {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::string device_name() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    return prop.name;
}

void matvec(const TensorInfo& t, std::span<const std::byte> bytes, const float* x, float* y) {
    if (!available()) throw std::runtime_error("CUDA backend requested but no CUDA device is available");
    if (t.shape.size() != 2) throw std::runtime_error("CUDA matvec requires rank-2 tensor: " + t.name);
    const uint64_t cols = t.shape[0];
    const uint64_t rows = t.shape[1];
    if ((t.type == TensorType::Q4_0 || t.type == TensorType::Q8_0) && cols % 32 != 0)
        throw std::runtime_error("CUDA quantized matvec requires columns divisible by 32: " + t.name);

    const void* dw = device_weight(bytes);
    scratch.reserve(static_cast<size_t>(cols), static_cast<size_t>(rows));
    check(cudaMemcpy(scratch.x, x, cols * sizeof(float), cudaMemcpyHostToDevice), "upload activation");

    constexpr unsigned threads = 256;
    const size_t shared = threads * sizeof(float);
    switch (t.type) {
        case TensorType::F16:
            matvec_f16_kernel<<<static_cast<unsigned>(rows), threads, shared>>>(
                static_cast<const __half*>(dw), scratch.x, scratch.y, cols);
            break;
        case TensorType::Q8_0:
            matvec_q8_kernel<<<static_cast<unsigned>(rows), threads, shared>>>(
                static_cast<const unsigned char*>(dw), scratch.x, scratch.y, cols);
            break;
        case TensorType::Q4_0:
            matvec_q4_kernel<<<static_cast<unsigned>(rows), threads, shared>>>(
                static_cast<const unsigned char*>(dw), scratch.x, scratch.y, cols);
            break;
        default:
            throw std::runtime_error("CUDA matvec tensor type not supported: " + t.name);
    }
    check(cudaGetLastError(), "launch matvec kernel");
    check(cudaMemcpy(y, scratch.y, rows * sizeof(float), cudaMemcpyDeviceToHost), "download activation");
}

} // namespace nedo::cuda_backend
