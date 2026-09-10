#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "nedo/gguf.hpp"
namespace nedo {
class SurfaceTokenizer {
public:
    explicit SurfaceTokenizer(const GgufFile& gguf);
    std::vector<uint32_t> encode(std::string_view text, bool add_bos=false) const;
    std::string decode(std::span<const uint32_t> ids) const;
    std::string token(uint32_t id) const;
    uint32_t vocab_size() const noexcept { return static_cast<uint32_t>(tokens_.size()); }
    std::optional<uint32_t> bos_id() const noexcept { return bos_; }
    std::optional<uint32_t> eos_id() const noexcept { return eos_; }
private:
    struct Node { int token{-1}; std::vector<std::pair<uint8_t,uint32_t>> next; };
    std::vector<std::string> tokens_;
    std::vector<Node> trie_;
    std::unordered_map<std::string,uint32_t> piece_to_id_;
    std::optional<uint32_t> bos_,eos_;
    static std::string unescape_rwkv(std::string_view s);
    void insert(uint32_t id, std::string_view bytes);
    void encode_bpe_segment(std::string_view bytes, std::vector<uint32_t>& out) const;
};
}
