#pragma once
#include "nedo/gguf.hpp"
#include "nedo/tokenizer.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace nedo {
struct ModelConfig {
    uint32_t vocab=0,dim=0,layers=0,heads=0,kv_heads=0,head_dim=0,ffn=0,context=0,sliding=0;
    uint32_t morph_layers=0,morph_shared=0,morph_root=0,morph_suffix=0;
    float rms_eps=1e-5f,rope_theta=10000.f;
};
struct GenerationConfig { uint32_t max_new_tokens=128; float temperature=0.8f,top_p=0.95f; uint32_t top_k=40; uint64_t seed=0; bool add_bos=false; };
struct SchemaReport { bool architecture_ok=false,standard_tensors_ok=false; uint32_t morph_tensors=0; std::vector<std::string> morph_names,missing,unmatched; std::string router_contract; };
class NedoModel {
public:
    explicit NedoModel(const std::filesystem::path& path);
    const ModelConfig& config()const noexcept{return cfg_;}
    const SchemaReport& schema()const noexcept{return schema_;}
    const GgufFile& gguf()const noexcept{return gguf_;}
    const SurfaceTokenizer& tokenizer()const noexcept{return tok_;}
    std::string summary()const;
    std::vector<uint32_t> tokenize(std::string_view s,bool bos=false)const{return tok_.encode(s,bos);}
    std::string detokenize(const std::vector<uint32_t>&v)const{return tok_.decode(v);}
    std::vector<uint32_t> generate_ids(const std::vector<uint32_t>& prompt_ids,const GenerationConfig& gc={});
    std::string generate(std::string_view prompt,const GenerationConfig& gc={});
private:
    struct Layer { const TensorInfo *an{},*q{},*k{},*v{},*o{},*fn{},*gate{},*up{},*down{}; };
    GgufFile gguf_; SurfaceTokenizer tok_; ModelConfig cfg_; SchemaReport schema_;
    const TensorInfo* emb_{}; const TensorInfo* out_norm_{}; const TensorInfo* out_{}; const TensorInfo* route_table_{};
    std::vector<Layer> blocks_;
    const TensorInfo* find_any(const std::vector<std::string>&names)const;
    void load_schema();
};
}
