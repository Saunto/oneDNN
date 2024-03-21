#include "cpu/rv64v/rv64_gemm_convolution_jit.hpp"
#include "common/utils.hpp"
#include "cpu/gemm_convolution_utils.hpp"
    
#include <cstdlib>
#include <iostream>
#include <vector>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::memory_tracking::names;

template <data_type_t T>
status_t
rv64_gemm_convolution_jit_fwd_t<T>::do_execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t* const, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const data_t* const, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t*, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t* const, DNNL_ARG_DST);
    auto const src_mb_size = pd()->IC() * pd()->IH() * pd()->IW();
    auto const dst_mb_size = pd()->OC() * pd()->OH() * pd()->OW();
    
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < pd()->MB(); ++n) {
        auto const pdst = &dst[n*dst_mb_size];
        auto const psrc = &src[n*src_mb_size];
        for(size_t i = 0; i < schedule.N; ++i)
            call_schedule(schedule, i, n, pdst, psrc, wei, bias);
    }
    return status::success;
}

template struct rv64_gemm_convolution_jit_fwd_t<data_type::f32>;


} // rv64
} // cpu
} // impl
} // dnnl