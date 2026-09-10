#include "nedo/tokenizer.hpp"
#include <algorithm>
#include <charconv>
#include <stdexcept>

namespace nedo {
namespace {
constexpr uint32_t kPad = 0;
constexpr uint32_t kBos = 1;
constexpr uint32_t kEos = 2;
constexpr uint32_t kByteBase = 3;
constexpr uint32_t kEntryBase = 259;

std::optional<uint32_t> meta_id(const GgufFile& g, std::string_view k) {
    if (auto v = g.u64_meta(k)) return static_cast<uint32_t>(*v);
    return {};
}

int hex_value(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
}

std::string SurfaceTokenizer::unescape_rwkv(std::string_view s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        // Also accept legacy <0xNN> byte-token spelling.
        if (i + 6 <= s.size() && s[i] == '<' && s[i + 1] == '0' && s[i + 2] == 'x' && s[i + 5] == '>') {
            const int hi = hex_value(s[i + 3]);
            const int lo = hex_value(s[i + 4]);
            if (hi >= 0 && lo >= 0) {
                o.push_back(static_cast<char>((hi << 4) | lo));
                i += 6;
                continue;
            }
        }
        if (s[i] != '\\') {
            o.push_back(s[i++]);
            continue;
        }
        // GGUF RWKV vocab stores arbitrary bytes as C-like escapes: \\xNN,
        // plus the usual newline/tab/carriage-return/backslash escapes.
        if (++i >= s.size()) {
            o.push_back('\\');
            break;
        }
        const char esc = s[i++];
        switch (esc) {
            case 'n': o.push_back('\n'); break;
            case 'r': o.push_back('\r'); break;
            case 't': o.push_back('\t'); break;
            case '\\': o.push_back('\\'); break;
            case 'x': {
                if (i + 1 < s.size()) {
                    const int hi = hex_value(s[i]);
                    const int lo = hex_value(s[i + 1]);
                    if (hi >= 0 && lo >= 0) {
                        o.push_back(static_cast<char>((hi << 4) | lo));
                        i += 2;
                        break;
                    }
                }
                // Malformed escape: preserve it losslessly rather than dropping data.
                o += "\\x";
                break;
            }
            default:
                // RWKV's reference unescaper treats an unknown escape as the escaped byte.
                o.push_back(esc);
                break;
        }
    }
    return o;
}

SurfaceTokenizer::SurfaceTokenizer(const GgufFile& g) {
    auto raw = g.string_array_meta("tokenizer.ggml.tokens");
    if (raw.empty()) throw std::runtime_error("GGUF has no tokenizer.ggml.tokens");
    if (raw.size() < kEntryBase) throw std::runtime_error("NDSRF surface vocab is smaller than fixed ID space");
    tokens_.reserve(raw.size()); trie_.push_back({});
    for (uint32_t i = 0; i < raw.size(); ++i) {
        tokens_.push_back(unescape_rwkv(raw[i]));
        if (i >= kEntryBase && !tokens_.back().empty()) insert(i, tokens_.back());
    }
    bos_ = meta_id(g, "tokenizer.ggml.bos_token_id").value_or(kBos);
    eos_ = meta_id(g, "tokenizer.ggml.eos_token_id").value_or(kEos);
}

void SurfaceTokenizer::insert(uint32_t id, std::string_view s) {
    uint32_t n = 0;
    for (unsigned char c : s) {
        auto& v = trie_[n].next;
        auto it = std::find_if(v.begin(), v.end(), [&](auto& p) { return p.first == c; });
        if (it == v.end()) { const uint32_t z = static_cast<uint32_t>(trie_.size()); v.emplace_back(c, z); trie_.push_back({}); n = z; }
        else n = it->second;
    }
    trie_[n].token = static_cast<int>(id);
}

std::vector<uint32_t> SurfaceTokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<uint32_t> out; out.reserve(text.size() + (add_bos ? 1 : 0));
    if (add_bos && bos_) out.push_back(*bos_);
    size_t pos = 0;
    while (pos < text.size()) {
        uint32_t n = 0; int best = -1; size_t best_end = pos;
        for (size_t j = pos; j < text.size(); ++j) {
            const uint8_t c = static_cast<uint8_t>(text[j]);
            const auto& v = trie_[n].next;
            auto it = std::find_if(v.begin(), v.end(), [&](const auto& p) { return p.first == c; });
            if (it == v.end()) break;
            n = it->second;
            if (trie_[n].token >= 0) { best = trie_[n].token; best_end = j + 1; }
        }
        if (best >= 0) { out.push_back(static_cast<uint32_t>(best)); pos = best_end; }
        else { out.push_back(kByteBase + static_cast<uint8_t>(text[pos])); ++pos; }
    }
    return out;
}

std::string SurfaceTokenizer::decode(std::span<const uint32_t> ids) const {
    std::string s;
    for (uint32_t id : ids) {
        if (id == kPad || id == kBos || id == kEos) continue;
        if (id >= kByteBase && id < kEntryBase) { s.push_back(static_cast<char>(id - kByteBase)); continue; }
        if (id >= tokens_.size()) throw std::runtime_error("token id out of range");
        s += tokens_[id];
    }
    return s;
}
std::string SurfaceTokenizer::token(uint32_t id) const { if (id >= tokens_.size()) return {}; return tokens_[id]; }
}
