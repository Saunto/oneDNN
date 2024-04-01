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
    
    int size = pd()->KH() * pd()->KW() * pd()->IC() * pd()->OH() * pd()->OW();

    // Allocate memory for the intermediate data
    // Using float col[size]; does not work due to stack size limitations
    // Using float* col = new float[size]; does work since it uses the heap  
    
    float* inter = new float[size](); 
    float* inter2 = new float[size]();
    
    call_schedule(schedule, 0, 0, dst, src, wei, bias, inter, inter2);

    delete[] inter;
    delete[] inter2;

    return status::success;
}

template struct rv64_gemm_convolution_jit_fwd_t<data_type::f32>;


} // rv64
} // cpu
} // impl
} // dnnl