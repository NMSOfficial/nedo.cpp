#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nedo {

enum class GgufType : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
    FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
};

enum class TensorType : uint32_t {
    F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8,
    Q8_1=9, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
    IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18, IQ1_S=19, IQ4_NL=20, IQ3_S=21,
    IQ2_S=22, IQ4_XS=23, I8=24, I16=25, I32=26, I64=27, F64=28,
    IQ1_M=29, BF16=30
};

using Scalar = std::variant<uint8_t,int8_t,uint16_t,int16_t,uint32_t,int32_t,float,bool,
                            uint64_t,int64_t,double,std::string>;
struct ArrayValue { GgufType type; std::vector<Scalar> values; };
using Value = std::variant<Scalar, ArrayValue>;

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    TensorType type{};
    uint64_t relative_offset{};
    uint64_t absolute_offset{};
    uint64_t n_elements{};
    uint64_t n_bytes{};
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    const std::byte* data() const noexcept { return data_; }
    uint64_t size() const noexcept { return size_; }
private:
    void close() noexcept;
    const std::byte* data_{};
    uint64_t size_{};
#ifdef _WIN32
    void* file_{};
    void* mapping_{};
#else
    int fd_{-1};
#endif
};

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path);
    uint32_t version() const noexcept { return version_; }
    uint64_t tensor_count() const noexcept { return tensors_.size(); }
    uint64_t metadata_count() const noexcept { return metadata_.size(); }
    uint64_t alignment() const noexcept { return alignment_; }
    uint64_t data_offset() const noexcept { return data_offset_; }
    const std::unordered_map<std::string,Value>& metadata() const noexcept { return metadata_; }
    const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
    const TensorInfo* tensor(std::string_view name) const noexcept;
    std::span<const std::byte> bytes(const TensorInfo& t) const;
    std::optional<std::string> string_meta(std::string_view key) const;
    std::optional<uint64_t> u64_meta(std::string_view key) const;
    std::optional<double> number_meta(std::string_view key) const;
    std::vector<std::string> string_array_meta(std::string_view key) const;
private:
    MappedFile file_;
    uint32_t version_{};
    uint64_t alignment_{32};
    uint64_t data_offset_{};
    std::unordered_map<std::string,Value> metadata_;
    std::vector<TensorInfo> tensors_;
};

uint64_t tensor_nbytes(TensorType type, uint64_t n_elements);
std::string tensor_type_name(TensorType type);
std::string value_to_string(const Value& v);

} // namespace nedo
