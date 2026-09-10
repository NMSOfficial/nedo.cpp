#include "nedo/model.hpp"
#include "nedo/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
namespace nedo {
namespace {
uint32_t getu(const GgufFile&g,std::initializer_list<const char*> ks,uint32_t def=0){for(auto k:ks)if(auto v=g.u64_meta(k))return(uint32_t)*v;return def;}
float getf(const GgufFile&g,std::initializer_list<const char*> ks,float def){for(auto k:ks)if(auto v=g.number_meta(k))return(float)*v;return def;}
std::string fmt(const char*pat,uint32_t i){char b[128];std::snprintf(b,sizeof(b),pat,i);return b;}
}
NedoModel::NedoModel(const std::filesystem::path&p):gguf_(p),tok_(gguf_){load_schema();}
const TensorInfo*NedoModel::find_any(const std::vector<std::string>&ns)const{for(auto&n:ns)if(auto*t=gguf_.tensor(n))return t;return nullptr;}
void NedoModel::load_schema(){
    auto arch=gguf_.string_meta("general.architecture").value_or(""); schema_.architecture_ok=(arch=="nedolm");
    cfg_.vocab=getu(gguf_,{"nedolm.vocab_size"},tok_.vocab_size()); cfg_.dim=getu(gguf_,{"nedolm.embedding_length","llama.embedding_length"},1536); cfg_.layers=getu(gguf_,{"nedolm.block_count","llama.block_count"},24);
    cfg_.heads=getu(gguf_,{"nedolm.attention.head_count","llama.attention.head_count"},12);cfg_.kv_heads=getu(gguf_,{"nedolm.attention.head_count_kv","llama.attention.head_count_kv"},4);cfg_.head_dim=cfg_.dim/cfg_.heads;
    cfg_.ffn=getu(gguf_,{"nedolm.feed_forward_length","llama.feed_forward_length"},5632);cfg_.context=getu(gguf_,{"nedolm.context_length","llama.context_length"},4096);cfg_.sliding=getu(gguf_,{"nedolm.attention.sliding_window","nedolm.sliding_window"},2048);
    cfg_.rms_eps=getf(gguf_,{"nedolm.attention.layer_norm_rms_epsilon","llama.attention.layer_norm_rms_epsilon"},1e-5f);cfg_.rope_theta=getf(gguf_,{"nedolm.rope.freq_base","llama.rope.freq_base"},10000.f);
    emb_=find_any({"token_embd.weight","tok_embeddings.weight","model.embed_tokens.weight"});out_norm_=find_any({"output_norm.weight","norm.weight"});out_=find_any({"output.weight","lm_head.weight"});if(!out_)out_=emb_;
    std::unordered_set<std::string> used;auto use=[&](const TensorInfo*t){if(t)used.insert(t->name);};use(emb_);use(out_norm_);use(out_);
    blocks_.resize(cfg_.layers); for(uint32_t i=0;i<cfg_.layers;++i){auto&b=blocks_[i];
        b.an=find_any({fmt("blk.%u.attn_norm.weight",i),fmt("layers.%u.attention_norm.weight",i)});b.q=find_any({fmt("blk.%u.attn_q.weight",i),fmt("layers.%u.attention.wq.weight",i)});b.k=find_any({fmt("blk.%u.attn_k.weight",i),fmt("layers.%u.attention.wk.weight",i)});b.v=find_any({fmt("blk.%u.attn_v.weight",i),fmt("layers.%u.attention.wv.weight",i)});b.o=find_any({fmt("blk.%u.attn_output.weight",i),fmt("layers.%u.attention.wo.weight",i)});
        b.fn=find_any({fmt("blk.%u.ffn_norm.weight",i),fmt("layers.%u.ffn_norm.weight",i)});b.gate=find_any({fmt("blk.%u.ffn_gate.weight",i),fmt("layers.%u.feed_forward.w1.weight",i)});b.down=find_any({fmt("blk.%u.ffn_down.weight",i),fmt("layers.%u.feed_forward.w2.weight",i)});b.up=find_any({fmt("blk.%u.ffn_up.weight",i),fmt("layers.%u.feed_forward.w3.weight",i)});
        std::vector<std::string> rc={fmt("blk.%u.ffn_route.weight",i),fmt("blk.%u.morph.weight",i),fmt("blk.%u.token_prior.weight",i),fmt("blk.%u.ffn_router.weight",i),fmt("layers.%u.morph_route",i)};b.route=find_any(rc);
        for(auto*t:{b.an,b.q,b.k,b.v,b.o,b.fn,b.gate,b.down,b.up})use(t);if(b.route){use(b.route);schema_.morph_names.push_back(b.route->name);schema_.morph_tensors++;}
        if(!b.an||!b.q||!b.k||!b.v||!b.o||!b.fn||!b.gate||!b.down||!b.up)schema_.missing.push_back("block "+std::to_string(i)+" standard tensor set");
    }
    if(!emb_)schema_.missing.push_back("token embedding");if(!out_norm_)schema_.missing.push_back("output norm");
    if(schema_.morph_tensors==0){for(auto&t:gguf_.tensors())if(t.shape.size()==1&&!used.count(t.name)){schema_.morph_names.push_back(t.name);schema_.morph_tensors++;}}
    for(auto&t:gguf_.tensors())if(!used.count(t.name)&&std::find(schema_.morph_names.begin(),schema_.morph_names.end(),t.name)==schema_.morph_names.end())schema_.unmatched.push_back(t.name);
    schema_.standard_tensors_ok=schema_.missing.empty();
    if(schema_.morph_tensors==18) schema_.router_contract="18 routing tensors discovered; exact token-prior equation requires exporter/runtime contract validation"; else schema_.router_contract="routing tensor set unresolved";
}
std::string NedoModel::summary()const{std::ostringstream o;o<<"NedoLM "<<cfg_.dim<<"d / "<<cfg_.layers<<"L / "<<cfg_.heads<<"Q:"<<cfg_.kv_heads<<"KV / FFN "<<cfg_.ffn<<" / ctx "<<cfg_.context<<" / sliding "<<cfg_.sliding<<"\n";o<<"GGUF v"<<gguf_.version()<<", "<<gguf_.tensor_count()<<" tensors, morph candidates="<<schema_.morph_tensors<<"\n";o<<"schema: "<<(schema_.architecture_ok?"nedolm":"unexpected architecture")<<", standard="<<(schema_.standard_tensors_ok?"ok":"missing")<<", router="<<schema_.router_contract;return o.str();}
std::string NedoModel::generate(std::string_view,const GenerationConfig&){
    throw std::runtime_error("nedo.cpp: exact NedoLM MorphFFN routing contract is not public/validated yet. Inspect model.schema / run nedo-inspect and provide the discovered routing tensor names+shapes before enabling generation. This guard prevents plausible-looking but mathematically incorrect output.");
}
}
