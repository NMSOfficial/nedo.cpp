#include "nedo/tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <limits>
#include <numeric>
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

bool word_byte(unsigned char c) noexcept {
    // UTF-8 continuation/lead bytes stay in the lexical word run. The production
    // tokenizer applies richer morphology cuts; this scanner preserves the same
    // hard whitespace/punctuation structure while BPE handles the surface bytes.
    return c >= 0x80 || std::isalnum(c) || c == '_';
}

bool white_byte(unsigned char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}
}

std::string SurfaceTokenizer::unescape_rwkv(std::string_view s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
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
                o += "\\x";
                break;
            }
            default: o.push_back(esc); break;
        }
    }
    return o;
}

SurfaceTokenizer::SurfaceTokenizer(const GgufFile& g) {
    auto raw = g.string_array_meta("tokenizer.ggml.tokens");
    if (raw.empty()) throw std::runtime_error("GGUF has no tokenizer.ggml.tokens");
    if (raw.size() < kEntryBase) throw std::runtime_error("NDSRF surface vocab is smaller than fixed ID space");
    tokens_.reserve(raw.size());
    piece_to_id_.reserve(raw.size() - kEntryBase);
    trie_.push_back({});
    for (uint32_t i = 0; i < raw.size(); ++i) {
        tokens_.push_back(unescape_rwkv(raw[i]));
        if (i >= kEntryBase && !tokens_.back().empty()) {
            insert(i, tokens_.back());
            piece_to_id_.emplace(tokens_.back(), i);
        }
    }
    bos_ = meta_id(g, "tokenizer.ggml.bos_token_id").value_or(kBos);
    eos_ = meta_id(g, "tokenizer.ggml.eos_token_id").value_or(kEos);
}

void SurfaceTokenizer::insert(uint32_t id, std::string_view s) {
    uint32_t n = 0;
    for (unsigned char c : s) {
        auto& v = trie_[n].next;
        auto it = std::find_if(v.begin(), v.end(), [&](auto& p) { return p.first == c; });
        if (it == v.end()) {
            const uint32_t z = static_cast<uint32_t>(trie_.size());
            v.emplace_back(c, z);
            trie_.push_back({});
            n = z;
        } else n = it->second;
    }
    trie_[n].token = static_cast<int>(id);
}

void SurfaceTokenizer::encode_bpe_segment(std::string_view bytes, std::vector<uint32_t>& out) const {
    if (bytes.empty()) return;
    if (bytes.size() == 1) {
        out.push_back(kByteBase + static_cast<unsigned char>(bytes[0]));
        return;
    }

    // NDSRF004 uses learned-entry ID order as merge rank. Start from exact bytes
    // and repeatedly remove the boundary whose adjacent pair has the lowest rank.
    std::vector<size_t> boundaries(bytes.size() + 1);
    std::iota(boundaries.begin(), boundaries.end(), size_t{0});
    while (boundaries.size() > 2) {
        uint32_t best_id = std::numeric_limits<uint32_t>::max();
        size_t best_boundary = 0;
        bool found = false;
        for (size_t i = 0; i + 2 < boundaries.size(); ++i) {
            const size_t left = boundaries[i];
            const size_t right = boundaries[i + 2];
            auto it = piece_to_id_.find(std::string(bytes.substr(left, right - left)));
            if (it != piece_to_id_.end() && it->second < best_id) {
                best_id = it->second;
                best_boundary = i + 1;
                found = true;
            }
        }
        if (!found) break;
        boundaries.erase(boundaries.begin() + static_cast<std::ptrdiff_t>(best_boundary));
    }

    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        const size_t left = boundaries[i];
        const size_t right = boundaries[i + 1];
        const size_t n = right - left;
        if (n == 1) {
            out.push_back(kByteBase + static_cast<unsigned char>(bytes[left]));
            continue;
        }
        auto it = piece_to_id_.find(std::string(bytes.substr(left, n)));
        if (it == piece_to_id_.end()) {
            // Defensive byte-exact fallback; a valid BPE merge chain should never land here.
            for (size_t p = left; p < right; ++p)
                out.push_back(kByteBase + static_cast<unsigned char>(bytes[p]));
        } else out.push_back(it->second);
    }
}

std::vector<uint32_t> SurfaceTokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<uint32_t> out;
    out.reserve(text.size() + (add_bos ? 1 : 0));
    if (add_bos && bos_) out.push_back(*bos_);

    size_t pos = 0;
    while (pos < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[pos]);

        if (c == ' ') {
            size_t ws_end = pos + 1;
            while (ws_end < text.size() && text[ws_end] == ' ') ++ws_end;
            // Production NDSRF004 allows one ordinary inter-word space to prefix
            // the following lexical root. Preserve that bridge here.
            if (ws_end == pos + 1 && ws_end < text.size() && word_byte(static_cast<unsigned char>(text[ws_end]))) {
                size_t end = ws_end + 1;
                while (end < text.size() && word_byte(static_cast<unsigned char>(text[end]))) ++end;
                encode_bpe_segment(text.substr(pos, end - pos), out);
                pos = end;
            } else {
                encode_bpe_segment(text.substr(pos, ws_end - pos), out);
                pos = ws_end;
            }
            continue;
        }

        if (white_byte(c)) {
            size_t end = pos + 1;
            while (end < text.size() && white_byte(static_cast<unsigned char>(text[end])) && text[end] != ' ') ++end;
            encode_bpe_segment(text.substr(pos, end - pos), out);
            pos = end;
            continue;
        }

        if (word_byte(c)) {
            size_t end = pos + 1;
            while (end < text.size() && word_byte(static_cast<unsigned char>(text[end]))) ++end;
            encode_bpe_segment(text.substr(pos, end - pos), out);
            pos = end;
            continue;
        }

        size_t end = pos + 1;
        while (end < text.size()) {
            const unsigned char x = static_cast<unsigned char>(text[end]);
            if (white_byte(x) || word_byte(x)) break;
            ++end;
        }
        encode_bpe_segment(text.substr(pos, end - pos), out);
        pos = end;
    }
    return out;
}

std::string SurfaceTokenizer::decode(std::span<const uint32_t> ids) const {
    std::string s;
    for (uint32_t id : ids) {
        if (id == kPad || id == kBos || id == kEos) continue;
        if (id >= kByteBase && id < kEntryBase) {
            s.push_back(static_cast<char>(id - kByteBase));
            continue;
        }
        if (id >= tokens_.size()) throw std::runtime_error("token id out of range");
        s += tokens_[id];
    }
    return s;
}

std::string SurfaceTokenizer::token(uint32_t id) const {
    if (id >= tokens_.size()) return {};
    return tokens_[id];
}
}
