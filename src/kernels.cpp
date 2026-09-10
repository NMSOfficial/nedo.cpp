#include "nedo/kernels.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#if defined(__AVX2__) || defined(_M_AVX2)
#define NEDO_X86_AVX2 1
#endif
#if defined(__F16C__) || defined(_M_AVX2)
#define NEDO_X86_F16C 1
#endif
#if defined(__FMA__) || defined(_M_AVX2)
#define NEDO_X86_FMA 1
#endif
#if defined(NEDO_X86_AVX2) || defined(NEDO_X86_F16C)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nedo::kernels {
float fp16_to_fp32(uint16_t h) noexcept {
#if defined(NEDO_X86_F16C)
    return _cvtsh_ss(h);
#else
    uint32_t s=(h&0x8000u)<<16, e=(h>>10)&0x1fu, m=h&0x3ffu, bits;
    if(e==0){ if(m==0) bits=s; else { e=1; while((m&0x400u)==0){m<<=1;--e;} m&=0x3ffu; bits=s+((e+112u)<<23)+(m<<13); } }
    else if(e==31) bits=s+0x7f800000u+(m<<13); else bits=s+((e+112u)<<23)+(m<<13);
    return std::bit_cast<float>(bits);
#endif
}
uint16_t fp32_to_fp16(float f) noexcept {
#if defined(NEDO_X86_F16C)
    return static_cast<uint16_t>(_cvtss_sh(f, 0));
#else
    uint32_t x=std::bit_cast<uint32_t>(f), s=(x>>16)&0x8000u; int e=int((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffffu;
    if(e<=0){ if(e<-10)return(uint16_t)s; m=(m|0x800000u)>>(1-e); return(uint16_t)(s+((m+0x1000u)>>13)); }
    if(e>=31)return(uint16_t)(s+0x7c00u); return(uint16_t)(s+(uint32_t(e)<<10)+((m+0x1000u)>>13));
#endif
}
void rms_norm(const float*x,const uint16_t*w,float*y,size_t n,float eps) noexcept{
    double ss=0.0; for(size_t i=0;i<n;++i)ss+=double(x[i])*x[i]; float inv=1.0f/std::sqrt(float(ss/n)+eps);
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
    for(size_t i=0;i<n;++i)y[i]=x[i]*inv*fp16_to_fp32(w[i]);
}
void rms_norm_tensor(const float*x,const TensorInfo&w,std::span<const std::byte>b,float*y,size_t n,float eps){
    if(w.shape.size()!=1 || w.shape[0]!=n) throw std::runtime_error("bad RMSNorm tensor: "+w.name);
    double ss=0.0; for(size_t i=0;i<n;++i) ss+=double(x[i])*x[i];
    const float inv=1.0f/std::sqrt(float(ss/n)+eps);
    if(w.type==TensorType::F32){
        const float* wf=reinterpret_cast<const float*>(b.data());
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
        for(size_t i=0;i<n;++i)y[i]=x[i]*inv*wf[i];
        return;
    }
    if(w.type==TensorType::F16){
        const uint16_t* wh=reinterpret_cast<const uint16_t*>(b.data());
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
        for(size_t i=0;i<n;++i)y[i]=x[i]*inv*fp16_to_fp32(wh[i]);
        return;
    }
    throw std::runtime_error("unsupported RMSNorm tensor type: "+w.name);
}
void add_inplace(float*d,const float*s,size_t n) noexcept{
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
    for(size_t i=0;i<n;++i)d[i]+=s[i];
}
void mul_inplace(float*d,const float*s,size_t n) noexcept{
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
    for(size_t i=0;i<n;++i)d[i]*=s[i];
}
void silu_mul(float*g,const float*u,size_t n) noexcept{
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp simd
#endif
    for(size_t i=0;i<n;++i){float v=g[i]; g[i]=(v/(1.0f+std::exp(-v)))*u[i];}
}
void softmax(float*x,size_t n) noexcept{ if(!n)return; float m=*std::max_element(x,x+n),s=0; for(size_t i=0;i<n;++i){x[i]=std::exp(x[i]-m);s+=x[i];} float inv=1.0f/s; for(size_t i=0;i<n;++i)x[i]*=inv; }
void rope(float*q,float*k,uint32_t nq,uint32_t nkv,uint32_t hd,uint64_t pos,float theta) noexcept{
    auto rot=[&](float*p,uint32_t nh){for(uint32_t h=0;h<nh;++h){float* v=p+h*hd;for(uint32_t i=0;i+1<hd;i+=2){float freq=std::pow(theta,-float(i)/float(hd));float a=float(pos)*freq,c=std::cos(a),s=std::sin(a);float x=v[i],y=v[i+1];v[i]=x*c-y*s;v[i+1]=x*s+y*c;}}}; rot(q,nq);rot(k,nkv);
}

namespace {
inline float dot_f16(const uint16_t*w,const float*x,size_t n){
    float s=0.f; size_t i=0;
#if defined(NEDO_X86_AVX2) && defined(NEDO_X86_F16C)
    __m256 acc=_mm256_setzero_ps();
    for(;i+8<=n;i+=8){__m128i h=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+i));__m256 wf=_mm256_cvtph_ps(h);__m256 xf=_mm256_loadu_ps(x+i);
#if defined(NEDO_X86_FMA)
        acc=_mm256_fmadd_ps(wf,xf,acc);
#else
        acc=_mm256_add_ps(acc,_mm256_mul_ps(wf,xf));
#endif
    }
    alignas(32) float tmp[8];_mm256_store_ps(tmp,acc);for(float v:tmp)s+=v;
#endif
    for(;i<n;++i)s+=fp16_to_fp32(w[i])*x[i]; return s;
}
inline float dot_q8(const std::byte*p,const float*x,size_t n){
    float sum=0; size_t nb=n/32; for(size_t b=0;b<nb;++b){uint16_t dh;std::memcpy(&dh,p,2);float d=fp16_to_fp32(dh);p+=2;auto q=reinterpret_cast<const int8_t*>(p);float s=0;
#if defined(NEDO_X86_AVX2)
        __m256 acc=_mm256_setzero_ps(); for(int i=0;i<32;i+=8){__m128i qi8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q+i));__m256i qi32=_mm256_cvtepi8_epi32(qi8);__m256 qf=_mm256_cvtepi32_ps(qi32);__m256 xf=_mm256_loadu_ps(x+b*32+i);
#if defined(NEDO_X86_FMA)
        acc=_mm256_fmadd_ps(qf,xf,acc);
#else
        acc=_mm256_add_ps(acc,_mm256_mul_ps(qf,xf));
#endif
        } alignas(32) float tmp[8];_mm256_store_ps(tmp,acc);for(float v:tmp)s+=v;
#else
        for(int i=0;i<32;++i)s+=float(q[i])*x[b*32+i];
#endif
        sum+=d*s;p+=32;}return sum;
}
inline float dot_q4(const std::byte*p,const float*x,size_t n){
    float sum=0; size_t nb=n/32; for(size_t b=0;b<nb;++b){uint16_t dh;std::memcpy(&dh,p,2);float d=fp16_to_fp32(dh);p+=2;auto q=reinterpret_cast<const uint8_t*>(p);float s=0;
#if defined(NEDO_X86_AVX2)
        __m128i qb=_mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
        __m128i mask=_mm_set1_epi8(0x0f), bias=_mm_set1_epi8(8);
        __m128i lo8=_mm_sub_epi8(_mm_and_si128(qb,mask),bias);
        __m128i hi8=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qb,4),mask),bias);
        __m256 acc=_mm256_setzero_ps();
        auto madd8=[&](const __m128i& z,const float* xv){__m256i zi=_mm256_cvtepi8_epi32(z);__m256 zf=_mm256_cvtepi32_ps(zi);__m256 xf=_mm256_loadu_ps(xv);
#if defined(NEDO_X86_FMA)
            acc=_mm256_fmadd_ps(zf,xf,acc);
#else
            acc=_mm256_add_ps(acc,_mm256_mul_ps(zf,xf));
#endif
        };
        madd8(lo8,x+b*32);madd8(_mm_srli_si128(lo8,8),x+b*32+8);madd8(hi8,x+b*32+16);madd8(_mm_srli_si128(hi8,8),x+b*32+24);
        alignas(32) float tmp[8];_mm256_store_ps(tmp,acc);for(float v:tmp)s+=v;
#else
        for(int i=0;i<16;++i){int lo=(q[i]&0x0f)-8,hi=(q[i]>>4)-8;s+=float(lo)*x[b*32+i]+float(hi)*x[b*32+i+16];}
#endif
        sum+=d*s;p+=16;}return sum;
}
}
void matvec(const TensorInfo&t,std::span<const std::byte>b,const float*x,float*y){
    if(t.shape.size()!=2)throw std::runtime_error("matvec requires rank-2 tensor: "+t.name); size_t cols=t.shape[0],rows=t.shape[1];
    uint64_t rowbytes=tensor_nbytes(t.type,cols);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows >= 256)
#endif
    for(int64_t r=0;r<(int64_t)rows;++r){const std::byte*p=b.data()+uint64_t(r)*rowbytes;switch(t.type){case TensorType::F16:y[r]=dot_f16(reinterpret_cast<const uint16_t*>(p),x,cols);break;case TensorType::Q8_0:y[r]=dot_q8(p,x,cols);break;case TensorType::Q4_0:y[r]=dot_q4(p,x,cols);break;default:throw std::runtime_error("matvec tensor type not supported");}}
}
void embedding_row(const TensorInfo&t,std::span<const std::byte>b,uint32_t row,float*out){
    if(t.shape.size()!=2||row>=t.shape[1])throw std::runtime_error("bad embedding row");size_t n=t.shape[0];uint64_t rb=tensor_nbytes(t.type,n);const std::byte*p=b.data()+uint64_t(row)*rb;
    if(t.type==TensorType::F16){auto*w=reinterpret_cast<const uint16_t*>(p);for(size_t i=0;i<n;++i)out[i]=fp16_to_fp32(w[i]);}
    else if(t.type==TensorType::Q8_0){for(size_t blk=0;blk<n/32;++blk){uint16_t dh;std::memcpy(&dh,p,2);float d=fp16_to_fp32(dh);p+=2;auto*q=reinterpret_cast<const int8_t*>(p);for(int i=0;i<32;++i)out[blk*32+i]=d*q[i];p+=32;}}
    else if(t.type==TensorType::Q4_0){for(size_t blk=0;blk<n/32;++blk){uint16_t dh;std::memcpy(&dh,p,2);float d=fp16_to_fp32(dh);p+=2;auto*q=reinterpret_cast<const uint8_t*>(p);for(int i=0;i<16;++i){out[blk*32+i]=d*((q[i]&15)-8);out[blk*32+i+16]=d*((q[i]>>4)-8);}p+=16;}}
    else throw std::runtime_error("embedding type unsupported");
}
}
