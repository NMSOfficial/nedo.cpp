#include "nedo/model.hpp"
#include "nedo/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace nedo {
namespace {
uint32_t getu(const GgufFile& g, std::initializer_list<const char*> ks, uint32_t def=0) {
    for (auto k : ks) if (auto v=g.u64_meta(k)) return static_cast<uint32_t>(*v);
    return def;
}
float getf(const GgufFile& g, std::initializer_list<const char*> ks, float def) {
    for (auto k : ks) if (auto v=g.number_meta(k)) return static_cast<float>(*v);
    return def;
}
std::string fmt(const char* pat,uint32_t i) {
    char b[128]; std::snprintf(b,sizeof(b),pat,i); return b;
}

float dot(const float* a, const float* b, uint32_t n) noexcept {
    double s=0.0;
    for (uint32_t i=0;i<n;++i) s += static_cast<double>(a[i])*b[i];
    return static_cast<float>(s);
}

uint32_t sample_token(const std::vector<float>& logits, const GenerationConfig& gc, std::mt19937_64& rng) {
    if (logits.empty()) throw std::runtime_error("empty logits");
    if (gc.temperature <= 0.f) {
        return static_cast<uint32_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    }
    std::vector<uint32_t> ids(logits.size());
    std::iota(ids.begin(),ids.end(),0u);
    const uint32_t k = gc.top_k ? std::min<uint32_t>(gc.top_k, static_cast<uint32_t>(ids.size())) : static_cast<uint32_t>(ids.size());
    if (k < ids.size()) {
        std::nth_element(ids.begin(),ids.begin()+k,ids.end(),[&](uint32_t a,uint32_t b){return logits[a]>logits[b];});
        ids.resize(k);
    }
    std::sort(ids.begin(),ids.end(),[&](uint32_t a,uint32_t b){return logits[a]>logits[b];});
    const float inv_t=1.f/gc.temperature;
    float mx=-std::numeric_limits<float>::infinity();
    for (auto id:ids) mx=std::max(mx,logits[id]*inv_t);
    std::vector<double> probs(ids.size());
    double sum=0.0;
    for (size_t i=0;i<ids.size();++i) { probs[i]=std::exp(double(logits[ids[i]]*inv_t-mx)); sum+=probs[i]; }
    for (double& p:probs) p/=sum;
    if (gc.top_p>0.f && gc.top_p<1.f) {
        double cum=0.0; size_t keep=0;
        for (;keep<probs.size();++keep) { cum+=probs[keep]; if (cum>=gc.top_p) { ++keep; break; } }
        keep=std::max<size_t>(1,std::min(keep,probs.size()));
        ids.resize(keep); probs.resize(keep);
        sum=std::accumulate(probs.begin(),probs.end(),0.0);
        for(double& p:probs)p/=sum;
    }
    std::discrete_distribution<size_t> dist(probs.begin(),probs.end());
    return ids[dist(rng)];
}
}

NedoModel::NedoModel(const std::filesystem::path& p):gguf_(p),tok_(gguf_){load_schema();}
const TensorInfo* NedoModel::find_any(const std::vector<std::string>& ns)const {
    for(auto& n:ns) if(auto* t=gguf_.tensor(n)) return t;
    return nullptr;
}

void NedoModel::load_schema(){
    const auto arch=gguf_.string_meta("general.architecture").value_or("");
    schema_.architecture_ok=(arch=="nedolm");
    cfg_.vocab=getu(gguf_,{"nedolm.vocab_size"},tok_.vocab_size());
    cfg_.dim=getu(gguf_,{"nedolm.embedding_length","llama.embedding_length"},1536);
    cfg_.layers=getu(gguf_,{"nedolm.block_count","llama.block_count"},24);
    cfg_.heads=getu(gguf_,{"nedolm.attention.head_count","llama.attention.head_count"},12);
    cfg_.kv_heads=getu(gguf_,{"nedolm.attention.head_count_kv","llama.attention.head_count_kv"},4);
    cfg_.head_dim=getu(gguf_,{"nedolm.rope.dimension_count"}, cfg_.heads ? cfg_.dim/cfg_.heads : 0);
    cfg_.ffn=getu(gguf_,{"nedolm.feed_forward_length","llama.feed_forward_length"},5632);
    cfg_.context=getu(gguf_,{"nedolm.context_length","llama.context_length"},4096);
    cfg_.sliding=getu(gguf_,{"nedolm.attention.sliding_window","nedolm.sliding_window"},2048);
    cfg_.morph_layers=getu(gguf_,{"nedolm.morph.layer_count"},0);
    cfg_.morph_shared=getu(gguf_,{"nedolm.morph.shared_width"},0);
    cfg_.morph_root=getu(gguf_,{"nedolm.morph.root_width"},0);
    cfg_.morph_suffix=getu(gguf_,{"nedolm.morph.suffix_width"},0);
    cfg_.rms_eps=getf(gguf_,{"nedolm.attention.layer_norm_rms_epsilon","llama.attention.layer_norm_rms_epsilon"},1e-5f);
    cfg_.rope_theta=getf(gguf_,{"nedolm.rope.freq_base","llama.rope.freq_base"},10000.f);

    emb_=find_any({"token_embd.weight","tok_embeddings.weight","model.embed_tokens.weight"});
    out_norm_=find_any({"output_norm.weight","norm.weight"});
    out_=find_any({"output.weight","lm_head.weight"});
    if(!out_)out_=emb_;
    route_table_=find_any({"blk.0.ffn_gate_tid2eid.weight"});

    std::unordered_set<std::string> used;
    auto use=[&](const TensorInfo*t){if(t)used.insert(t->name);};
    use(emb_); use(out_norm_); use(out_); use(route_table_);

    blocks_.resize(cfg_.layers);
    for(uint32_t i=0;i<cfg_.layers;++i){
        auto& b=blocks_[i];
        b.an=find_any({fmt("blk.%u.attn_norm.weight",i),fmt("layers.%u.attention_norm.weight",i)});
        b.q=find_any({fmt("blk.%u.attn_q.weight",i),fmt("layers.%u.attention.wq.weight",i)});
        b.k=find_any({fmt("blk.%u.attn_k.weight",i),fmt("layers.%u.attention.wk.weight",i)});
        b.v=find_any({fmt("blk.%u.attn_v.weight",i),fmt("layers.%u.attention.wv.weight",i)});
        b.o=find_any({fmt("blk.%u.attn_output.weight",i),fmt("layers.%u.attention.wo.weight",i)});
        b.fn=find_any({fmt("blk.%u.ffn_norm.weight",i),fmt("layers.%u.ffn_norm.weight",i)});
        b.gate=find_any({fmt("blk.%u.ffn_gate.weight",i),fmt("layers.%u.feed_forward.w1.weight",i)});
        b.down=find_any({fmt("blk.%u.ffn_down.weight",i),fmt("layers.%u.feed_forward.w2.weight",i)});
        b.up=find_any({fmt("blk.%u.ffn_up.weight",i),fmt("layers.%u.feed_forward.w3.weight",i)});
        for(auto*t:{b.an,b.q,b.k,b.v,b.o,b.fn,b.gate,b.down,b.up}) use(t);
        if(!b.an||!b.q||!b.k||!b.v||!b.o||!b.fn||!b.gate||!b.down||!b.up)
            schema_.missing.push_back("block "+std::to_string(i)+" standard tensor set");
    }
    if(!emb_)schema_.missing.push_back("token embedding");
    if(!out_norm_)schema_.missing.push_back("output norm");

    if(cfg_.morph_layers){
        if(!route_table_) schema_.missing.push_back("blk.0.ffn_gate_tid2eid.weight");
        else {
            schema_.morph_tensors=1;
            schema_.morph_names.push_back(route_table_->name);
            const bool route_shape=route_table_->shape.size()==2 && route_table_->shape[0]==2 && route_table_->shape[1]==cfg_.vocab;
            if(!route_shape || route_table_->type!=TensorType::F32)
                schema_.missing.push_back("MorphFFN token-id routing table shape/type [2,vocab] F32");
        }
        if(cfg_.morph_shared+cfg_.morph_root+cfg_.morph_suffix!=cfg_.ffn)
            schema_.missing.push_back("MorphFFN widths do not sum to feed_forward_length");
        if(cfg_.morph_layers>cfg_.layers) schema_.missing.push_back("MorphFFN layer count exceeds block count");
    }

    for(auto&t:gguf_.tensors()) if(!used.count(t.name)) schema_.unmatched.push_back(t.name);
    schema_.standard_tensors_ok=schema_.missing.empty();
    if(route_table_ && cfg_.morph_shared+cfg_.morph_root+cfg_.morph_suffix==cfg_.ffn) {
        schema_.router_contract="tid2eid ROOT/SUFFIX table; FFN=shared+root+suffix";
    } else if(!cfg_.morph_layers) {
        schema_.router_contract="standard FFN";
    } else {
        schema_.router_contract="MorphFFN contract invalid";
    }
}

std::string NedoModel::summary()const {
    std::ostringstream o;
    o<<"NedoLM "<<cfg_.dim<<"d / "<<cfg_.layers<<"L / "<<cfg_.heads<<"Q:"<<cfg_.kv_heads
     <<"KV / FFN "<<cfg_.ffn<<" / ctx "<<cfg_.context<<" / sliding "<<cfg_.sliding<<"\n";
    o<<"GGUF v"<<gguf_.version()<<", "<<gguf_.tensor_count()<<" tensors";
    if(cfg_.morph_layers) o<<", MorphFFN "<<cfg_.morph_layers<<"L ("<<cfg_.morph_shared<<" shared + "<<cfg_.morph_root<<" root + "<<cfg_.morph_suffix<<" suffix)";
    o<<"\n";
    o<<"schema: "<<(schema_.architecture_ok?"nedolm":"unexpected architecture")
     <<", standard="<<(schema_.standard_tensors_ok?"ok":"missing")<<", router="<<schema_.router_contract;
    return o.str();
}

std::vector<uint32_t> NedoModel::generate_ids(const std::vector<uint32_t>& prompt_ids,const GenerationConfig& gc,const std::function<void(uint32_t)>& on_token){
    if(!schema_.architecture_ok) throw std::runtime_error("GGUF architecture is not nedolm");
    if(!schema_.standard_tensors_ok) {
        std::ostringstream e; e<<"model schema is incomplete";
        for(const auto& m:schema_.missing)e<<"; "<<m;
        throw std::runtime_error(e.str());
    }
    if(!cfg_.heads || !cfg_.kv_heads || cfg_.heads%cfg_.kv_heads || cfg_.dim!=cfg_.heads*cfg_.head_dim)
        throw std::runtime_error("invalid GQA dimensions");
    if(!cfg_.context) throw std::runtime_error("context_length is zero");

    std::vector<uint32_t> tokens=prompt_ids;
    const uint32_t bos=tok_.bos_id().value_or(1u);
    if(gc.add_bos && (tokens.empty() || tokens.front()!=bos)) tokens.insert(tokens.begin(),bos);
    if(tokens.empty()) tokens.push_back(bos);
    for(uint32_t id:tokens) if(id>=cfg_.vocab) throw std::runtime_error("prompt token id out of range");
    if(tokens.size()>cfg_.context) {
        tokens.erase(tokens.begin(),tokens.end()-cfg_.context);
    }

    const uint32_t D=cfg_.dim, H=cfg_.heads, KH=cfg_.kv_heads, HD=cfg_.head_dim;
    const uint32_t KVD=KH*HD, F=cfg_.ffn;
    const uint32_t W=std::max<uint32_t>(1,std::min(cfg_.sliding?cfg_.sliding:cfg_.context,cfg_.context));
    const uint32_t q_per_kv=H/KH;
    const float inv_sqrt_hd=1.f/std::sqrt(static_cast<float>(HD));

    // F32 KV cache favors reference correctness. A packed F16 cache can be added as a backend optimization.
    const size_t cache_layer_stride=static_cast<size_t>(W)*KVD;
    std::vector<float> kcache(static_cast<size_t>(cfg_.layers)*cache_layer_stride,0.f);
    std::vector<float> vcache(static_cast<size_t>(cfg_.layers)*cache_layer_stride,0.f);

    std::vector<float> x(D),xn(D),q(H*HD),k(KVD),v(KVD),att(H*HD),tmp(D);
    std::vector<float> gate(F),up(F),logits(cfg_.vocab),scores(W);

    std::span<const std::byte> route_bytes;
    const float* route=nullptr;
    if(route_table_) {
        route_bytes=gguf_.bytes(*route_table_);
        route=reinterpret_cast<const float*>(route_bytes.data());
    }

    auto apply_morph=[&](uint32_t token_id,uint32_t layer){
        if(layer < cfg_.layers - cfg_.morph_layers || !route) return;
        const float root=route[static_cast<size_t>(token_id)*2];
        const float suffix=route[static_cast<size_t>(token_id)*2+1];
        const uint32_t s=cfg_.morph_shared;
        const uint32_t r_end=s+cfg_.morph_root;
        const uint32_t z_end=r_end+cfg_.morph_suffix;
        // NedoLM keeps the first (layers-morph_layers) blocks on the standard FFN path;
        // MorphFFN occupies the trailing blocks. Neurons are [shared | root | suffix].
        // The tid2eid pair is [ROOT, SUFFIX]; OTHER is (0,0). Shared is always active.
        if(root<0.5f) std::fill(gate.begin()+s,gate.begin()+r_end,0.f);
        if(suffix<0.5f) std::fill(gate.begin()+r_end,gate.begin()+z_end,0.f);
    };

    auto forward=[&](uint32_t token_id,uint64_t pos){
        kernels::embedding_row(*emb_,gguf_.bytes(*emb_),token_id,x.data());
        for(uint32_t li=0;li<cfg_.layers;++li){
            const auto& b=blocks_[li];
            kernels::rms_norm_tensor(x.data(),*b.an,gguf_.bytes(*b.an),xn.data(),D,cfg_.rms_eps);
            kernels::matvec(*b.q,gguf_.bytes(*b.q),xn.data(),q.data());
            kernels::matvec(*b.k,gguf_.bytes(*b.k),xn.data(),k.data());
            kernels::matvec(*b.v,gguf_.bytes(*b.v),xn.data(),v.data());
            kernels::rope(q.data(),k.data(),H,KH,HD,pos,cfg_.rope_theta);

            const uint32_t slot=static_cast<uint32_t>(pos%W);
            float* kc=kcache.data()+static_cast<size_t>(li)*cache_layer_stride+static_cast<size_t>(slot)*KVD;
            float* vc=vcache.data()+static_cast<size_t>(li)*cache_layer_stride+static_cast<size_t>(slot)*KVD;
            std::copy(k.begin(),k.end(),kc); std::copy(v.begin(),v.end(),vc);

            std::fill(att.begin(),att.end(),0.f);
            const uint64_t start=(pos+1>W)?pos+1-W:0;
            const uint32_t nctx=static_cast<uint32_t>(pos-start+1);
            for(uint32_t h=0;h<H;++h){
                const uint32_t kh=h/q_per_kv;
                const float* qh=q.data()+static_cast<size_t>(h)*HD;
                for(uint32_t j=0;j<nctx;++j){
                    const uint64_t p=start+j; const uint32_t ps=static_cast<uint32_t>(p%W);
                    const float* kk=kcache.data()+static_cast<size_t>(li)*cache_layer_stride+static_cast<size_t>(ps)*KVD+static_cast<size_t>(kh)*HD;
                    scores[j]=dot(qh,kk,HD)*inv_sqrt_hd;
                }
                kernels::softmax(scores.data(),nctx);
                float* ah=att.data()+static_cast<size_t>(h)*HD;
                for(uint32_t j=0;j<nctx;++j){
                    const uint64_t p=start+j; const uint32_t ps=static_cast<uint32_t>(p%W);
                    const float* vv=vcache.data()+static_cast<size_t>(li)*cache_layer_stride+static_cast<size_t>(ps)*KVD+static_cast<size_t>(kh)*HD;
                    const float a=scores[j];
                    for(uint32_t d=0;d<HD;++d) ah[d]+=a*vv[d];
                }
            }
            kernels::matvec(*b.o,gguf_.bytes(*b.o),att.data(),tmp.data());
            kernels::add_inplace(x.data(),tmp.data(),D);

            kernels::rms_norm_tensor(x.data(),*b.fn,gguf_.bytes(*b.fn),xn.data(),D,cfg_.rms_eps);
            kernels::matvec(*b.gate,gguf_.bytes(*b.gate),xn.data(),gate.data());
            kernels::matvec(*b.up,gguf_.bytes(*b.up),xn.data(),up.data());
            kernels::silu_mul(gate.data(),up.data(),F);
            apply_morph(token_id,li);
            kernels::matvec(*b.down,gguf_.bytes(*b.down),gate.data(),tmp.data());
            kernels::add_inplace(x.data(),tmp.data(),D);
        }
        kernels::rms_norm_tensor(x.data(),*out_norm_,gguf_.bytes(*out_norm_),xn.data(),D,cfg_.rms_eps);
        kernels::matvec(*out_,gguf_.bytes(*out_),xn.data(),logits.data());
    };

    uint64_t pos=0;
    for(uint32_t id:tokens){
        if(pos>=cfg_.context) break;
        forward(id,pos++);
    }

    const uint64_t seed=gc.seed?gc.seed:std::random_device{}();
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> generated;
    generated.reserve(gc.max_new_tokens);
    const uint32_t eos=tok_.eos_id().value_or(2u);
    for(uint32_t n=0;n<gc.max_new_tokens && pos<cfg_.context;++n){
        const uint32_t next=sample_token(logits,gc,rng);
        if(next==eos) break;
        generated.push_back(next);
        if(on_token) on_token(next);
        forward(next,pos++);
    }
    return generated;
}

std::string NedoModel::generate(std::string_view prompt,const GenerationConfig& gc){
    const auto prompt_ids=tok_.encode(prompt,false);
    return tok_.decode(generate_ids(prompt_ids,gc));
}
}
