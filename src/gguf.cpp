#include "nedo/gguf.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nedo {

MappedFile::MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) { file_ = nullptr; throw std::runtime_error("cannot open GGUF"); }
    LARGE_INTEGER n{}; if (!GetFileSizeEx(file_, &n)) { close(); throw std::runtime_error("GetFileSizeEx failed"); }
    size_ = static_cast<uint64_t>(n.QuadPart);
    mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_) { close(); throw std::runtime_error("CreateFileMapping failed"); }
    data_ = static_cast<const std::byte*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
    if (!data_) { close(); throw std::runtime_error("MapViewOfFile failed"); }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open GGUF: " + path.string());
    struct stat st{}; if (fstat(fd_, &st) != 0) { close(); throw std::runtime_error("fstat failed"); }
    size_ = static_cast<uint64_t>(st.st_size);
    void* p = mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { data_ = nullptr; close(); throw std::runtime_error("mmap failed"); }
    data_ = static_cast<const std::byte*>(p);
#ifdef MADV_RANDOM
    madvise(const_cast<std::byte*>(data_), static_cast<size_t>(size_), MADV_RANDOM);
#endif
#endif
}
MappedFile::~MappedFile(){ close(); }
MappedFile::MappedFile(MappedFile&& o) noexcept { *this = std::move(o); }
MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this == &o) return *this; close(); data_=o.data_; size_=o.size_; o.data_=nullptr; o.size_=0;
#ifdef _WIN32
    file_=o.file_; mapping_=o.mapping_; o.file_=nullptr; o.mapping_=nullptr;
#else
    fd_=o.fd_; o.fd_=-1;
#endif
    return *this;
}
void MappedFile::close() noexcept {
#ifdef _WIN32
    if(data_) UnmapViewOfFile(data_); if(mapping_) CloseHandle(mapping_); if(file_) CloseHandle(file_);
    data_=nullptr; mapping_=nullptr; file_=nullptr;
#else
    if(data_) munmap(const_cast<std::byte*>(data_), static_cast<size_t>(size_)); if(fd_>=0) ::close(fd_);
    data_=nullptr; fd_=-1;
#endif
    size_=0;
}

namespace {
class Cursor {
public:
    explicit Cursor(const MappedFile& f): p_(f.data()), begin_(f.data()), end_(f.data()+f.size()) {}
    template<class T> T read() {
        if (static_cast<uint64_t>(end_-p_) < sizeof(T)) throw std::runtime_error("truncated GGUF");
        T v; std::memcpy(&v,p_,sizeof(T)); p_ += sizeof(T);
        static_assert(std::endian::native == std::endian::little, "nedo.cpp currently supports little-endian targets");
        return v;
    }
    std::string str() {
        uint64_t n=read<uint64_t>(); if(n > static_cast<uint64_t>(end_-p_)) throw std::runtime_error("invalid GGUF string");
        std::string s(reinterpret_cast<const char*>(p_), static_cast<size_t>(n)); p_+=n; return s;
    }
    uint64_t offset() const { return static_cast<uint64_t>(p_-begin_); }
private: const std::byte* p_; const std::byte* begin_; const std::byte* end_;
};

Scalar read_scalar(Cursor& c, GgufType t) {
    switch(t){
        case GgufType::UINT8:return c.read<uint8_t>(); case GgufType::INT8:return c.read<int8_t>();
        case GgufType::UINT16:return c.read<uint16_t>(); case GgufType::INT16:return c.read<int16_t>();
        case GgufType::UINT32:return c.read<uint32_t>(); case GgufType::INT32:return c.read<int32_t>();
        case GgufType::FLOAT32:return c.read<float>(); case GgufType::BOOL:return bool(c.read<uint8_t>());
        case GgufType::STRING:return c.str(); case GgufType::UINT64:return c.read<uint64_t>();
        case GgufType::INT64:return c.read<int64_t>(); case GgufType::FLOAT64:return c.read<double>();
        default: throw std::runtime_error("invalid scalar GGUF type");
    }
}
Value read_value(Cursor& c, GgufType t){
    if(t != GgufType::ARRAY) return Scalar(read_scalar(c,t));
    auto et=static_cast<GgufType>(c.read<uint32_t>()); uint64_t n=c.read<uint64_t>();
    if(et==GgufType::ARRAY) throw std::runtime_error("nested GGUF arrays unsupported by spec/runtime");
    if(n > 100000000ull) throw std::runtime_error("unreasonable GGUF array");
    ArrayValue a{et,{}}; a.values.reserve(static_cast<size_t>(n));
    for(uint64_t i=0;i<n;++i) a.values.push_back(read_scalar(c,et)); return a;
}
uint64_t align_up(uint64_t n,uint64_t a){ return (n+a-1)/a*a; }
}

uint64_t tensor_nbytes(TensorType type, uint64_t n){
    switch(type){
        case TensorType::F32: return n*4; case TensorType::F16: return n*2; case TensorType::BF16:return n*2;
        case TensorType::Q4_0: if(n%32) throw std::runtime_error("Q4_0 elements not block aligned"); return (n/32)*18;
        case TensorType::Q8_0: if(n%32) throw std::runtime_error("Q8_0 elements not block aligned"); return (n/32)*34;
        case TensorType::I8:return n; case TensorType::I16:return n*2; case TensorType::I32:return n*4; case TensorType::I64:return n*8;
        default: throw std::runtime_error("tensor type not implemented: " + std::to_string(static_cast<uint32_t>(type)));
    }
}
std::string tensor_type_name(TensorType t){
    switch(t){case TensorType::F32:return"F32";case TensorType::F16:return"F16";case TensorType::Q4_0:return"Q4_0";case TensorType::Q8_0:return"Q8_0";case TensorType::BF16:return"BF16";default:return"type("+std::to_string((uint32_t)t)+")";}
}

GgufFile::GgufFile(const std::filesystem::path& path):file_(path){
    Cursor c(file_); uint32_t magic=c.read<uint32_t>();
    if(magic != 0x46554747u) throw std::runtime_error("not a GGUF file (bad magic)");
    version_=c.read<uint32_t>(); if(version_<2 || version_>3) throw std::runtime_error("unsupported GGUF version");
    uint64_t nt=c.read<uint64_t>(), nkv=c.read<uint64_t>();
    if(nt>1000000 || nkv>1000000) throw std::runtime_error("invalid GGUF counts");
    metadata_.reserve(static_cast<size_t>(nkv));
    for(uint64_t i=0;i<nkv;++i){ auto k=c.str(); auto t=static_cast<GgufType>(c.read<uint32_t>()); metadata_.emplace(std::move(k),read_value(c,t)); }
    if(auto a=u64_meta("general.alignment")) alignment_=*a;
    if(alignment_==0 || (alignment_&(alignment_-1))) throw std::runtime_error("GGUF alignment must be power-of-two");
    tensors_.reserve(static_cast<size_t>(nt));
    for(uint64_t i=0;i<nt;++i){
        TensorInfo ti; ti.name=c.str(); uint32_t nd=c.read<uint32_t>(); if(nd==0||nd>4) throw std::runtime_error("invalid tensor rank");
        ti.shape.resize(nd); ti.n_elements=1; for(uint32_t d=0;d<nd;++d){ ti.shape[d]=c.read<uint64_t>(); if(ti.shape[d]==0 || ti.n_elements>UINT64_MAX/ti.shape[d]) throw std::runtime_error("invalid tensor shape"); ti.n_elements*=ti.shape[d]; }
        ti.type=static_cast<TensorType>(c.read<uint32_t>()); ti.relative_offset=c.read<uint64_t>(); ti.n_bytes=tensor_nbytes(ti.type,ti.n_elements); tensors_.push_back(std::move(ti));
    }
    data_offset_=align_up(c.offset(),alignment_);
    for(auto& t:tensors_){ t.absolute_offset=data_offset_+t.relative_offset; if(t.absolute_offset>file_.size() || t.n_bytes>file_.size()-t.absolute_offset) throw std::runtime_error("tensor outside GGUF file: "+t.name); }
}
const TensorInfo* GgufFile::tensor(std::string_view n) const noexcept { for(auto& t:tensors_) if(t.name==n) return &t; return nullptr; }
std::span<const std::byte> GgufFile::bytes(const TensorInfo& t) const { return {file_.data()+t.absolute_offset,static_cast<size_t>(t.n_bytes)}; }
std::optional<std::string> GgufFile::string_meta(std::string_view k) const {
    auto it=metadata_.find(std::string(k)); if(it==metadata_.end()) return{};
    if(auto s=std::get_if<Scalar>(&it->second)) if(auto p=std::get_if<std::string>(s)) return *p; return{};
}
std::optional<uint64_t> GgufFile::u64_meta(std::string_view k) const {
    auto it=metadata_.find(std::string(k)); if(it==metadata_.end()) return{}; auto s=std::get_if<Scalar>(&it->second); if(!s)return{};
    return std::visit([](auto&&v)->std::optional<uint64_t>{using T=std::decay_t<decltype(v)>;if constexpr(std::is_integral_v<T>)return static_cast<uint64_t>(v);else return{};},*s);
}
std::optional<double> GgufFile::number_meta(std::string_view k) const {
    auto it=metadata_.find(std::string(k)); if(it==metadata_.end()) return{}; auto s=std::get_if<Scalar>(&it->second); if(!s)return{};
    return std::visit([](auto&&v)->std::optional<double>{using T=std::decay_t<decltype(v)>;if constexpr(std::is_arithmetic_v<T>)return static_cast<double>(v);else return{};},*s);
}
std::vector<std::string> GgufFile::string_array_meta(std::string_view k) const {
    std::vector<std::string> out; auto it=metadata_.find(std::string(k)); if(it==metadata_.end())return out; auto a=std::get_if<ArrayValue>(&it->second); if(!a||a->type!=GgufType::STRING)return out;
    out.reserve(a->values.size()); for(auto&v:a->values) if(auto p=std::get_if<std::string>(&v)) out.push_back(*p); return out;
}
std::string value_to_string(const Value& v){
    if(auto a=std::get_if<ArrayValue>(&v)) return "array["+std::to_string(a->values.size())+"]";
    auto s=std::get_if<Scalar>(&v); if(!s)return{}; return std::visit([](auto&&x){using T=std::decay_t<decltype(x)>;if constexpr(std::is_same_v<T,std::string>)return x;else if constexpr(std::is_same_v<T,bool>)return std::string(x?"true":"false");else return std::to_string(x);},*s);
}
}
