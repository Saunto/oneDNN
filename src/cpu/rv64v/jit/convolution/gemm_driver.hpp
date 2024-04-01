#ifndef CPU_RV64V_GEMM_JIT_DRIVER_HPP
#define CPU_RV64V_GEMM_JIT_DRIVER_HPP

#include "cpu/rv64v/rvjit/rvjit.hpp"
#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm {

struct jit_convolution_configuration_t {
    // Convolutional arguments
    int mb;                // The convolution minibatch size
    int oh;                // The destination tensor height
    int ow;                // The destination tensor width
    int ih;                // The source tensor height
    int iw;                // The source tensor width
    int kh;                // The weights tensor height
    int kw;                // The weights tensor width
    int oc;                // The number of output feature maps
    int ic;                // The number of input feature maps
    int ngroups;           // The number of feature map groups (unsupported)
    prop_kind_t prop_kind; // The convolution propagation direction
    data_type_t src_dt;    // The source tensor data type
    data_type_t wei_dt;    // The weight tensor data type
    data_type_t dst_dt;    // The destination tensor data type
    data_type_t bias_dt;   // The bias vector data type (unsupported)
    bool with_bias;        // Convolution uses bias or not (true unsupported)
    int stride_h;          // The input tensor stride over the IH dimension
    int stride_w;          // The input tensor stride over the IW dimension
    int l_pad;             // Zero-padding at the input tensor left side
    int t_pad;             // Zero-padding at the input tensor top side
    int dilate_h;          // The weights dilatation over the H dimension
    int dilate_w;          // The weights dilatation over the W dimension
    // Hardware Parameters
    int maxvl;                   // The hardware maximum vector length
    int vcores;                  // Number of vector cores
    int vcore_shared_cache_size; // The VPU core shared cache size, usually LLC
    int l1d_cache_size;          // The L1 data cache size
    int l1d_cache_line_size;     // The L1 data cache line size
    int nvregs;                  // The number of ISA named vector registers
    // Software Optimizations
    int icb;   /// Activation tensor (src) blocking factor over the IC dim
    int ocb;   /// Activation tensor (dst) blocking factor over the OC dim
    int w_icb; /// Weights tensor (wei) blocking factor over the IC dim
    int w_ocb; /// Weights tensor (wei) blocking factor over the OC dim
    int w_dim_permute[4];  /// The weight tensor dimension order
    int w_inner_blocks[2]; /// The weight tensor inner block sizes
    int w_inner_idx[2];    /// The weight tensor inner block indices
    bool vdim_is_oc;       /// Whether the vectorized loop is OC or IC
    int vlen; /// The vector length configuration used within the kernel
    int rbw;  /// Unroll factor for the output tensor width loop
    int rbc;  /// Unroll factor for the output channel loop
    int k_c;  /// Loop size over the non-vectorized channel dim
    int k_h;  /// Loop size over the H dim walking the vector source tensor
    int k_w;  /// Loop size over the W dim walking the vector source tensor
};

struct convolution_schedule_t {
    struct jit_conv_kernel_args_t {
        const void *dst;
        const void *src;
        const void *wei;
        const void *bias;
        const void *inter;
        const void *inter2;
        size_t vlen;
        size_t h_loop_size;
        size_t w_loop_size;
        size_t load_partials;
        //size_t bwdd_zrow;
    };

    struct precalculated_args {
        size_t dst;
        size_t src;
        size_t wei;
        size_t bias;
        size_t vlen;
        size_t load_partials;
        //size_t bwdd_zrow;
        size_t h_loop_size;
        size_t w_loop_size;
    };

    using pkernel_t = void (*)(jit_conv_kernel_args_t&);

    jit_convolution_configuration_t cfg; // Seed configuration struct
    pkernel_t *calls;                    // Sequence of kernels to calls
    precalculated_args *args;            // Sequence of kernels arguments
    size_t N;                            // Sequence size
    rvjit::function* handles[32];        // JIT ukernels handles
    size_t NJ;                           // Number of JIT ukernels handles
};

bool
init_conf(jit_convolution_configuration_t &cfg, const convolution_desc_t &cd,
          const memory_desc_t &dst_md, const memory_desc_t &src_md,
          const memory_desc_t &wei_md, const memory_desc_t &bias_md);

bool pick_memory_formats_from_conf(const jit_convolution_configuration_t &cfg,
                                   memory_desc_t &dst_md,
                                   memory_desc_t &src_md,
                                   memory_desc_t &wei_md,
                                   memory_desc_t &bias_md);

void init_schedule(convolution_schedule_t &s,
                    const jit_convolution_configuration_t &acfg);
void free_schedule(convolution_schedule_t &s);
void call_schedule(const convolution_schedule_t &s, int i, int mb,
                    const float *dst, const float *src, const float *wei, const float *bias, const float *inter, const float *inter2);

} // namespace gemm
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_DRIVER_HPP