#include <stddef.h>
#include <iostream>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "common/type_helpers.hpp"
#include "cpu/rv64v/jit/convolution/gemm_kernel.hpp"


namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm{

using namespace rvjit;
using jit_conv_kernel_args_t = convolution_schedule_t::jit_conv_kernel_args_t;

namespace {
// The following code is based on oneDNN/src/cpu/gemm/f32/gemm_utils_f32.cpp
using namespace dnnl::impl::utils;

// Uses m as a unroll factor.
// Purpose: Copies the rows or columns (depending on isTransA) of matrix A
// into the workspace ws, to make the data access pattern more cache friendly.
template <typename data_t>
void copy_A(bool isTransA, dim_t K, const data_t *A, const dim_t lda, data_t *ws, const int m) {
    for (dim_t k = 0; k < K; k++) {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < m; i++) {
            ws[i] = isTransA ? A[i * lda + k] : A[i + k * lda];
        }
        ws += m;
    }
}

// Uses m and n as unroll factors.
// Purpose: Performs the core computation of the matrix multiplication.
// The function computes the matrix product of A and B and accumulates the
// result into matrix C. 
template <typename data_t, bool isTransA, bool isTransB>
void kernel_mxn(dim_t K, const data_t *A, const dim_t lda, const data_t *B,
        const dim_t ldb, data_t *C, const dim_t ldc, const data_t alpha,
        const data_t beta, const dim_t m, const dim_t n) {
    data_t c[m*n] = {static_cast<data_t>(0.)};
    for (dim_t k = 0; k < K; k++) {
        for (dim_t j = 0; j < n; j++) {
            data_t b = isTransB ? B[j + k * ldb] : B[k + j * ldb];
            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < m; i++) {
                data_t a = isTransA ? A[i * lda + k] : A[i + lda * k];
                c[i + m * j] += a * b;
            }
        }
    }
    for (dim_t j = 0; j < n; j++) {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < m; i++) {
            C[i + j * ldc] = (beta == static_cast<data_t>(0.))
                    ? alpha * c[i + m * j]
                    : alpha * c[i + m * j]
                            + beta * C[i + j * ldc];
        }
    }
}

// Uses m and n as unroll factors.
// Purpose: Computes the block matrix multiplication of A and B and accumulates
// the result into matrix using kernel_mxn function. 
template <typename data_t, bool isTransA, bool isTransB>
void block_ker(const dim_t M, const dim_t N, const dim_t K, const data_t *A,
        const dim_t lda, const data_t *B, const dim_t ldb, data_t *C,
        const dim_t ldc, const data_t alpha, const data_t beta, data_t *ws,
        bool do_copy, const dim_t m, const dim_t n) {
    dim_t Nu = rnd_dn(N, n);
    dim_t Mu = rnd_dn(M, m);
    for (dim_t i = 0; i < Mu; i += m) {
        for (dim_t j = 0; j < Nu; j += n) {
            const data_t *b = isTransB ? &B[j] : &B[j * ldb];
            const data_t *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) { copy_A<data_t>(isTransA, K, a, lda, ws); }
                kernel_mxn<data_t, false, isTransB>(K, ws,
                        m, b, ldb, &C[i + j * ldc], ldc,
                        alpha, beta);
            } else {
                kernel_mxn<data_t, isTransA, isTransB>(
                        K, a, lda, b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            }
        }
    }
    // tail processing
    for (dim_t i = 0; i < M; i++) {
        for (dim_t j = Nu; j < N; j++) {
            data_t c = beta == static_cast<data_t>(0.) ? static_cast<data_t>(0.)
                                                       : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                data_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                data_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (dim_t i = Mu; i < M; i++) {
        for (dim_t j = 0; j < Nu; j++) {
            data_t c = beta == static_cast<data_t>(0.) ? static_cast<data_t>(0.)
                                                       : beta * C[i + j * ldc];
            for (dim_t p = 0; p < K; p++) {
                data_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                data_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

// Purpose: The top-level function for the GEMM computation. The function
// It partitions the input matrices into blocks and calls the block_ker function
template <typename data_t, bool isTransA, bool isTransB>
void gemm_ithr(const dim_t M, const dim_t N, const dim_t K, const data_t alpha,
        const data_t *A, const dim_t lda, const data_t *B, const dim_t ldb,
        const data_t beta, data_t *C, const dim_t ldc, bool do_copy,
        data_t *ws, const dim_t BM, const dim_t BN, const dim_t BK) {

    const data_t *curA;
    const data_t *curB;
    data_t *curC;

    if ((M <= 0) || (N <= 0)) return;

    if ((K <= 0) || (alpha == static_cast<data_t>(0))) {
        dim_t MN = N * M;
        if (beta == static_cast<data_t>(0.)) {
            for (dim_t j = 0; j < MN; j++)
                C[j] = static_cast<data_t>(0.);
        } else if (beta != static_cast<data_t>(1.)) {
            for (dim_t j = 0; j < MN; j++)
                C[j] *= beta;
        }
        return;
    }

    for (dim_t Bk = 0; Bk < K; Bk += BK) {
        dim_t kb = nstl::min(K - Bk, BK);
        for (dim_t Bm = 0; Bm < M; Bm += BM) {
            dim_t mb = nstl::min(M - Bm, BM);
            for (dim_t Bn = 0; Bn < N; Bn += BN) {
                dim_t nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;
                if (Bk == 0) {
                    block_ker<data_t, isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha, beta, ws, do_copy);
                } else {
                    block_ker<data_t, isTransA, isTransB>(mb, nb, kb, curA, lda,
                            curB, ldb, curC, ldc, alpha,
                            static_cast<data_t>(1.0), ws, do_copy);
                }
            }
        }
    }
}
} // namespace



void jit_convolution_kernel_t::code() {
    const int nvregs = traits.erbw * traits.erbc;
    const size_t wei_sew = types::data_type_size(cfg.wei_dt);
    const size_t bia_sew = cfg.with_bias ? types::data_type_size(cfg.bias_dt) : 0;
    const size_t src_sew = types::data_type_size(cfg.src_dt);
    const size_t dst_sew = types::data_type_size(cfg.dst_dt);

    const size_t out_sew =
        utils::pick_by_prop_kind(cfg.prop_kind, dst_sew, src_sew, wei_sew);

    /// Offset to output pointer field in kernel args structure
    const size_t args_out_ptr = utils::pick_by_prop_kind(cfg.prop_kind,
                                offsetof(jit_conv_kernel_args_t, dst),
                                offsetof(jit_conv_kernel_args_t, src),
                                offsetof(jit_conv_kernel_args_t, wei));
    /// Output tensor W dimension stride
    const size_t w_off = utils::pick_by_prop_kind(cfg.prop_kind,
                                dst_sew * cfg.ocb,
                                src_sew * cfg.icb * cfg.stride_w,
                                wei_sew * cfg.w_ocb * cfg.w_icb);
    /// Output tensor C inter-block dimension stride
    const size_t c_off = utils::pick_by_prop_kind(cfg.prop_kind,
                                dst_sew * cfg.ocb * cfg.oh * cfg.ow,
                                src_sew * cfg.icb * cfg.ih * cfg.iw,
                                wei_sew * cfg.vlen);
    /// Output register block
    vr_t vout[32];
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);
    /// Pool of available caller-saved general purpose registers
    register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1});

    /// Move data from/to output vector registers and tensor memory region
    const auto move_outputs = [&](bool is_load, register_pool_t pool) {
        const gpr_t out = pool.pick();
        assembly_constant_t asm_c_off;
        assembly_constant_t asm_w_off;


        asm_w_off = asm_const(pool, w_off);
        // Subtract the accumulated W offset
        asm_c_off = asm_const(pool, c_off - (cfg.rbw-1) * w_off);

        ld(out, a0, args_out_ptr);
        if (cfg.rbc > 1)
            prepare_constant(asm_c_off);
        if (cfg.rbw > 1)
            prepare_constant(asm_w_off);

        int w = 0, c = 0;
        bool is_done = false;

        do {
            const auto id = w * cfg.rbc + c;
            if (is_load)
                vl(vout[id], out, out_sew);
            else
                vs(vout[id], out, out_sew);
            //if (!is_bwdw) {
            is_done = utils::nd_iterator_step(c, cfg.rbc, w, cfg.rbw);
            if (!is_done) {
                if (w)
                    add_constant(out, out, asm_w_off);
                else
                    add_constant(out, out, asm_c_off);
            }
        } while(!is_done);
    };

    // Initialization Segment
    do {
        const int sew = utils::pick_by_prop_kind(cfg.prop_kind,
            dst_sew, src_sew, wei_sew);
        const gpr_t vlen = tmp_pool.pick();
        const gpr_t load_partials = tmp_pool.pick();
        ld(vlen, a0, offsetof(jit_conv_kernel_args_t, vlen));
        vsetvli(x0, vlen, vsew(sew) | vlmul(1));
        lw(load_partials, a0, offsetof(jit_conv_kernel_args_t, load_partials));
        bnez(load_partials, "load_psum");
        for (int i = 0; i < nvregs; ++i)
            vxor_vv(vout[i], vout[i], vout[i]);
        j("compute");
        L("load_psum");
        move_outputs(true, tmp_pool);
    } while (0);

    // Compute Segment
    tmp_pool.reset();
    L("compute");
    switch (cfg.prop_kind) {
        case prop_kind::forward: {
            fwdd_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        case prop_kind::forward_inference: {
            fwdd_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        default:
            assert(!"unsupported propagation kind");
    }
        
    // Store Partial Sums Segment
    // TODO: this is temporary BIAS support
    if (cfg.with_bias) {//&& is_fwdd) {
        auto tmp   = tmp_pool.pick();
        auto bias_ptr  = tmp_pool.head();
        auto vbias = static_cast<vr_t>(nvregs);

        ld(tmp, a0, offsetof(jit_conv_kernel_args_t, load_partials));
        bnez(tmp, "store_psum");
        ld(bias_ptr, a0, offsetof(jit_conv_kernel_args_t, bias));
        vl(vbias, bias_ptr, bia_sew);
        for (int i = 0; i < nvregs; ++i)
            vfadd_vv(vout[i], vout[i], vbias);
    }
    tmp_pool.reset();
    L("store_psum");
    move_outputs(false, tmp_pool);
    ret();
}

int
starting_point(jit_convolution_configuration_t cfg, int rbw, int kh, int kw) {
    const int dst_sew = types::data_type_size(cfg.dst_dt);
    const int src_sew = types::data_type_size(cfg.src_dt);
    switch (cfg.prop_kind) {
        case prop_kind::forward_inference:
        case prop_kind::forward_training:
            return (kh * (cfg.dilate_h+1) * cfg.iw
                 + kw * (cfg.dilate_w+1) + rbw * cfg.stride_w) * cfg.icb * src_sew;
        case prop_kind::backward_data:
            return (kh * cfg.ow + kw + rbw) * cfg.ocb * dst_sew;
        default:
            assert(false);
            return 0;
    }
}

void jit_convolution_kernel_t::fwdd_inner_loops(rvjit::vr_t *vout, int rb_sz, register_pool_t &rp) {
    // --------------- Number of iterations for compute loops -----------------
    // The following constitutes the micro-kernel compute loops
    // for (icb = 0; icb < min(icb, k_c) / icb; ++icb)
    //  for (wic = 0; wic < min(icb, k_c) / w_icb; ++wic) // only pointwise
    //   for (kh = padTop; kh < k_h - padBot; ++kh)
    //    for (kw = 0; kh < k_w; ++kw)
    //     for (ic = 0; ic < w_icb; ++ic)
    auto const xcb = cfg.icb;
    auto const wxcb = cfg.w_icb;
    auto const kxcb_loop_sz = nstl::max(1, cfg.k_c/xcb);
    auto const xcb_loop_sz  = (cfg.k_c > xcb ? xcb : cfg.k_c) / wxcb;
    const int wei_sew = types::data_type_size(cfg.wei_dt);
    const int src_sew = types::data_type_size(cfg.src_dt);

    // ---------------------------- Tensor Offsets ----------------------------
    const int kw_off   = cfg.w_icb * cfg.w_ocb * wei_sew;
    const int xw_off   = cfg.stride_w * xcb    * src_sew;
    const int wxcb_off = wxcb                  * src_sew;
    int xcb_off        = cfg.ih * cfg.iw * xcb * src_sew;

    // ---------------------- Code generation conditions ----------------------
    /// A pointwise kernel does not need two loops to cover the ICB block
    const bool c_is_pointwise = cfg.k_h * cfg.k_w == 1;
    /// Generate w_icb loop as icb > w_icb
    const bool c_gen_icb_loop = xcb_loop_sz > 1 && !c_is_pointwise;
    /// Walk the register block applying immediate offsets (save on adds)
    const bool c_use_imm_rbw = imm_range / 2 > xw_off;

    // ------------------------------ Registers -------------------------------
    const gpr_t vsrc = rp.pick();
    const gpr_t xsrc = rp.pick();
    const gpr_t ptr_bcast = rp.pick();  // Working pointer to src tensor
    const gpr_t xc_off = rp.pick();     // Accumulated ic offset within IC blk
    const gpr_t xc_off_max = rp.pick(); // Max ic offset within IC blk (use in 'beq`)
    const gpr_t i_kxc = rp.pick();      // k_ic loop iterator (sub-tensors)
    const gpr_t i_xcb = rp.pick();      // icb loop iterator (number of w_icb blocks)
    auto wei_off = asm_const(rp, cfg.vlen * wei_sew);
    auto rbw_off = asm_const(rp, traits.erbw == 1 ? 0 :
        c_use_imm_rbw ? imm_range : xw_off);
    const gpr_t rbw_start = rp.pick();  // Pointer to the first activation
    const gpr_t tmp = rp.pick();        // Multi-purpose temporary across the ukernel

    /// VFMA source tensor scalar operand (implicit vector broadcast)
    const fpr_t f_bcast[4] = {ft0, ft1, ft2, ft3};
    static constexpr unsigned int nf_bcast = 4;
    unsigned int f_bcast_id = 0;
    /// VFMA weights tensor vector operand
    const vr_t vwei = static_cast<vr_t>(rb_sz);

    // --------------------- Control variables and lambdas --------------------
    /// Update the src pointer to the next convolution window (out register)
    /// @return true when the update requires an add instruction
    /// @details If the rbw_off is an immediate, the function only returns true
    /// when the accumulated offset overflows the 12bit representation range.
    auto should_issue_add_to_update_rbw_offset = [&](int &off) {
        if (c_use_imm_rbw) {
            off += xw_off;
            if (can_be_imm12(off))
                return false;
            off -= imm_range;
        }
        return true;
    };

    // -------------------------- Compute kernel begin ------------------------
    ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, wei));
    ld(xsrc, a0, offsetof(jit_conv_kernel_args_t, src));
    prepare_constant(wei_off);
    load_constant(xc_off_max, (c_is_pointwise ? xcb_loop_sz : 1) * wxcb_off);
    if (traits.erbw > 1)    prepare_constant(rbw_off);
    if (kxcb_loop_sz > 1)   load_constant(i_kxc, kxcb_loop_sz);
    L("kxc_loop");
    if (c_gen_icb_loop)     load_constant(i_xcb, xcb_loop_sz);
    L("xcb_loop");

    int wei_skip = traits.rbpadT * cfg.kw;
    for (int kh = traits.rbpadT; kh < cfg.k_h - traits.rbpadB; ++kh) {
        for (int kw = 0; kw < cfg.k_w; ++kw) {
            /// The first vout index this loop after padding mask
            const int rw_s = traits.rbpadL > kw ? traits.rbpadL - kw : 0;
            /// The last vout index this loop after padding mask
            const int rw_e = traits.erbw - (traits.rbpadR >= cfg.k_w-kw
                ? (traits.rbpadR - (cfg.k_w - (kw+1))) : 0);

            if (rw_e - rw_s < 1) {
                ++wei_skip;
                continue;
            } else if (wei_skip) {
                // Adjust the weight ptr to account for padding
                add_constant(vsrc, vsrc, tmp, wei_skip * kw_off);
                wei_skip = 0;
            }

            // Reset the intra-block channel offset, with range: [0,w_icb)
            li(xc_off, 0);

            // Adjust the working source pointer to the first activation point
            /// The offset from the start of the register the block
            int rbw_imm = (c_use_imm_rbw && traits.erbw > 1) ? imm_min() : 0;
            /// The offset for the starting point of the first output register
            int spatial_offset = starting_point(cfg, rw_s, kh, kw);
            add_constant(rbw_start, xsrc, tmp, spatial_offset - rbw_imm);

            // Start the inner-most loop over w_icb for this point (kh,kw)
            char local_label[16]; /// The current label name
            snprintf(local_label, 16, "l%d", kh * cfg.k_w + kw);
            L(local_label);

            // Set the src ptr to the IC offset at the current sub-tensor
            add(ptr_bcast, rbw_start, xc_off);

            // Load the next vector operand
            vl(vwei, vsrc, wei_sew);
            add_constant(vsrc, vsrc, wei_off);

            // Reuse the vector operand across the output register block
            flw(f_bcast[f_bcast_id], ptr_bcast, rbw_imm);
            vfmacc_vf(vout[rw_s], f_bcast[f_bcast_id], vwei);
            f_bcast_id = (f_bcast_id + 1) % nf_bcast;
            for (int rw = rw_s+1; rw < rw_e; rw++) {
                if (should_issue_add_to_update_rbw_offset(rbw_imm))
                    add_constant(ptr_bcast, ptr_bcast, rbw_off);
                flw(f_bcast[f_bcast_id], ptr_bcast, rbw_imm);
                vfmacc_vf(vout[rw], f_bcast[f_bcast_id], vwei);
                f_bcast_id = (f_bcast_id + 1) % nf_bcast;
            }

            // Loop end: one iteration over the non-vectorized dim block
            addi(xc_off, xc_off, src_sew);
            bne(xc_off, xc_off_max, local_label);
        }
        wei_skip += cfg.kw - cfg.k_w;
    }
    wei_skip += (cfg.kh + traits.rbpadB - cfg.k_h) * cfg.kw;

    if (c_gen_icb_loop || kxcb_loop_sz > 1) {
        // Adjust the weight ptr to account for masked computations due to pad
        if (wei_skip) {
            add_constant(vsrc, vsrc, tmp, wei_skip * kw_off);
            wei_skip = 0;
        }
        // Adjust the src pointer to account for computed channels 
        add(xsrc, xsrc, xc_off_max);
        xcb_off -= (xcb_loop_sz * wxcb_off); // remove accum increments to src
    }

    // Loop end: completed a w_icb segment of the IC block
    if (c_gen_icb_loop) {
        addi(i_xcb, i_xcb, -1);
        bnez(i_xcb, "xcb_loop");
    }

    // Loop end: computed the IC block, go to next sub-tensor
    if (kxcb_loop_sz > 1) {
        add_constant(xsrc, xsrc, tmp, xcb_off);
        addi(i_kxc, i_kxc, -1);
        bnez(i_kxc, "kxc_loop");
    }
}


} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv