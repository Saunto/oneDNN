#ifndef CPU_RV64V_JIT_CONVOLUTION_KERNEL_FWDD_HPP
#define CPU_RV64V_JIT_CONVOLUTION_KERNEL_FWDD_HPP

#include "cpu/rv64v/jit/jit_assembler.hpp"
#include "cpu/rv64v/jit/convolution/gemm_driver.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm {

/// A structure to identify a micro-kernel solving a sub-convolution
/// Rationale: The convolution micro-kernel computes a convolution over a
/// sub-tensor with shape dictated by the register block optimization across
/// the spatial domain.
/// For a given convolution problem, the sub-tensor shape may not segment the
/// spatial domain in equal parts.
/// The JIT driver must create one JIT functions for each sub-tensor shape
/// (register block optimization) to solve the convolution problem.
/// The following structure serve to uniquely identify a micro-kernel from a
/// set of micro-kernels generated with the same set of configurations,
/// enabling to associate it with a sub-tensor.
/// @details
/// The defining characteristic is the effective register block shape for the
/// micro-kernel, and wheter or not this shape overlaps the tensor padding,
/// as the latter condition might cause some computations to be skipped.
struct kernel_traits_t {
    int erbw; // Effective register block width
    int erbc; // Effective register block channel size
    int rbpadT; // H axis padding overlap to the top of the register block
    int rbpadB; // H axis padding overlap to the bot of the register block
    int rbpadR; // W axis padding overlap to the right of the register block
    int rbpadL; // W axis padding overlap to the left of the register block

    bool operator ==(const kernel_traits_t &o) {
        return erbw == o.erbw && erbc == o.erbc
            && rbpadT == o.rbpadT && rbpadB == o.rbpadB
            && rbpadR == o.rbpadR && rbpadL == o.rbpadL;
    }
};

struct jit_convolution_kernel_t : public jit_assembler {
private:
    jit_convolution_configuration_t cfg;
    kernel_traits_t traits;

public:
    jit_convolution_kernel_t(const jit_convolution_configuration_t &c,
                                  const kernel_traits_t &t)
        : jit_assembler(), cfg(c), traits(t) {}
    void code(convolution_schedule_t::jit_conv_kernel_args_t kargs);

private:
    const int imm_range = imm12_max() - imm_min();

    void im2col_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int channels,  int height,  int width,int ksize,  int stride, int pad);
    void col2im_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int channels, int height, int width, int ksize, int stride, int pad);
    void gemm_nn_noalpha(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,int M, int N, int K, float ALPHA, int lda, int ldb, int ldc);
    void gemm_nn_original(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int M, int N, int K, float ALPHA, int A_offset, int lda, int B_offset, int ldb, int C_offset, int ldc);
    void gemm_nn_noalpha_unroll163loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int M, int N, int K, float ALPHA, int A_offset, int lda, int B_offset, int ldb, int C_offset, int ldc, int unroll);

    void blocked_gemm(convolution_schedule_t::jit_conv_kernel_args_t kargs, rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int M, int N, int K, float ALPHA,
        int lda, int ldb, int ldc, int block_size_M, int block_size_N, int block_size_K);

    void gemm_cpu(convolution_schedule_t::jit_conv_kernel_args_t kargs, rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,int TA, int TB, int M, int N, int K, float ALPHA, int lda, int ldb, float BETA, int ldc);

    template <typename data_t> dnnl_status_t gemm(const char *transa_, const char *transb_,
    const dim_t *M_, const dim_t *N_, const dim_t *K_, const data_t *alpha_,
    const data_t *A, const dim_t *lda_, const data_t *B, const dim_t *ldb_,
    const data_t *beta_, data_t *C, const dim_t *ldc_, const data_t *bias);
};

} // namespace gemm
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_JIT_CONVOLUTION_KERNEL_FWDD_HPP