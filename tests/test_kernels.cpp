#include "nedo/kernels.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
bool near(float a, float b, float eps = 1e-4f) { return std::abs(a - b) <= eps; }

int test_half_and_vector_ops() {
    for (float x : {0.f, 1.f, -2.f, 0.333f, 65504.f}) {
        auto h = nedo::kernels::fp32_to_fp16(x);
        float y = nedo::kernels::fp16_to_fp32(h);
        float tol = std::max(1e-3f, std::abs(x) * 2e-3f);
        if (std::abs(x - y) > tol) return 1;
    }
    float a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
    nedo::kernels::add_inplace(a, b, 3);
    return (a[0] == 5 && a[2] == 9) ? 0 : 2;
}

int test_f32_rmsnorm() {
    nedo::TensorInfo t{"norm", {2}, nedo::TensorType::F32};
    float wf[2] = {2.f, 0.5f};
    std::vector<std::byte> b(sizeof(wf));
    std::memcpy(b.data(), wf, sizeof(wf));
    float x[2] = {3.f, 4.f}, y[2]{};
    nedo::kernels::rms_norm_tensor(x, t, b, y, 2, 0.f);
    const float inv = 1.f/std::sqrt(12.5f);
    return (near(y[0], 3.f*inv*2.f) && near(y[1], 4.f*inv*0.5f)) ? 0 : 6;
}

int test_q8_matvec() {
    nedo::TensorInfo t{"q8", {32, 1}, nedo::TensorType::Q8_0};
    std::vector<std::byte> w(34);
    const uint16_t one = nedo::kernels::fp32_to_fp16(1.f);
    std::memcpy(w.data(), &one, 2);
    for (int i = 0; i < 32; ++i) w[2 + i] = std::byte(static_cast<unsigned char>(static_cast<int8_t>(i - 16)));
    float x[32];
    for (float& v : x) v = 1.f;
    float y = 0.f;
    nedo::kernels::matvec(t, w, x, &y);
    return near(y, -16.f) ? 0 : 3;
}

int test_q4_matvec_and_embedding() {
    nedo::TensorInfo t{"q4", {32, 1}, nedo::TensorType::Q4_0};
    std::vector<std::byte> w(18);
    const uint16_t one = nedo::kernels::fp32_to_fp16(1.f);
    std::memcpy(w.data(), &one, 2);
    // Low nibbles 0..15 => -8..7. High nibbles 15..0 => 7..-8.
    for (int i = 0; i < 16; ++i) w[2 + i] = std::byte(static_cast<unsigned char>(((15 - i) << 4) | i));
    float x[32];
    for (float& v : x) v = 1.f;
    float y = 0.f;
    nedo::kernels::matvec(t, w, x, &y);
    if (!near(y, -16.f)) return 4;

    float row[32]{};
    nedo::kernels::embedding_row(t, w, 0, row);
    for (int i = 0; i < 16; ++i) {
        if (!near(row[i], float(i - 8)) || !near(row[i + 16], float(7 - i))) return 5;
    }
    return 0;
}
}

int main() {
    if (int rc = test_half_and_vector_ops()) return rc;
    if (int rc = test_f32_rmsnorm()) return rc;
    if (int rc = test_q8_matvec()) return rc;
    if (int rc = test_q4_matvec_and_embedding()) return rc;
    std::cout << "ok\n";
    return 0;
}
