#include <stddef.h>
#include <iostream>

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

void jit_convolution_kernel_t::code() {
    const int nvregs = traits.erbw * traits.erbc;
    const size_t wei_sew = types::data_type_size(cfg.wei_dt);
    const size_t bia_sew = cfg.with_bias ? types::data_type_size(cfg.bias_dt) : 0;
    const size_t src_sew = types::data_type_size(cfg.src_dt);
    const size_t dst_sew = types::data_type_size(cfg.dst_dt);
    const bool is_bwdw = cfg.prop_kind == prop_kind::backward_weights;
    const bool is_bwdd = cfg.prop_kind == prop_kind::backward_data;
    const bool is_fwdd = utils::one_of(cfg.prop_kind,
        prop_kind::forward_inference, prop_kind::forward_training);

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

        if (!is_bwdw) {
            asm_w_off = asm_const(pool, w_off);
            // Subtract the accumulated W offset
            asm_c_off = asm_const(pool, c_off - (cfg.rbw-1) * w_off);
        } else {
            asm_c_off = asm_const(pool, c_off);
            // Subtract the accumulated C offset
            asm_w_off = asm_const(pool, w_off - (cfg.rbc-1) * c_off);
        }   

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
            if (!is_bwdw) {
                is_done = utils::nd_iterator_step(c, cfg.rbc, w, cfg.rbw);
                if (!is_done) {
                    if (w)
                        add_constant(out, out, asm_w_off);
                    else
                        add_constant(out, out, asm_c_off);
                }
            } else {
                is_done = utils::nd_iterator_step(w, cfg.rbw, c, cfg.rbc);
                if (!is_done) {
                    if (c)
                        add_constant(out, out, asm_c_off);
                    else
                        add_constant(out, out, asm_w_off);
                }
            }
        } while(!is_done);
    };

    const auto bwdd_strided_zero_elems = [&](register_pool_t &pool) {
        /// Offset of the bwdw_zero flag in the ukernel input structure
        const size_t args_out_bwdd_zero =
            offsetof(jit_conv_kernel_args_t, bwdd_zrow);
        /// A vector register initialized with zeroes
        const vr_t vz = vout[0];
        /// Number of elements to zero in the first row per register
        const int zew = (cfg.stride_w - 1) * cfg.icb;
        /// Number of elements to zero in each of the subsequent rows
        const int zeh = cfg.stride_w * traits.erbw * cfg.icb;

        // Check if there are any elements to set to zero
        if (cfg.stride_h * cfg.stride_w == 1)
            return;

        /// Pointer to the base of the output tensor
        const gpr_t out = pool.pick();

        // Zero elements in the first row skipped due to horizontal stride
        if (zew) {
            auto blk_size        = asm_const(pool, 1 * cfg.icb * src_sew);
            auto skip_blk_stride = asm_const(pool, 2 * cfg.icb * src_sew);

            // Move to the first block with elements to set to zero
            prepare_constant(blk_size);
            prepare_constant(skip_blk_stride);
            ld(out, a0, args_out_ptr);
            add_constant(out, out, blk_size);
            // Set each activation block to zero
            for (int p = 0; p < traits.erbw; ++p) {
                vs(vz, out, out_sew);
                for (int i = 2; i < cfg.stride_w; ++i) {
                    add_constant(out, out, blk_size);
                    vs(vz, out, out_sew);
                }
                if (p+1 < traits.erbw)
                    add_constant(out, out, skip_blk_stride);
            }
        }
        // Zero elements in subsequent rows skipped due to vertical stride
        if (zeh) {
            /// Loop to zero elements in the same row
            const bool needs_col_loop = zeh > cfg.maxvl;
            /// Loop to cover all rows skipped due to vertical stride
            const bool needs_row_loop = cfg.stride_h > 2;
            /// Stride to advance to the next row (walk the IH dimension)
            auto h_stride = asm_const(pool, cfg.iw * cfg.icb * src_sew);
            /// The loop iterator over the rows (used when needs_row_loop)
            gpr_t row_loop_it = x0;
            /// The loop iterator over the row columns
            gpr_t col_loop_it = pool.pick();

            // Check the function arguments to see if this must be performed
            lw(pool.head(), a0, args_out_bwdd_zero);
            beqz(pool.head(), "compute");

            // Prepare the output pointer and the stride register
            ld(out, a0, args_out_ptr);
            prepare_constant(h_stride);

            // Set up loop iterator for rows when stride skips 2 or more rows
            if (needs_row_loop) {
                row_loop_it = pool.pick();
                load_constant(row_loop_it, cfg.stride_h-1);
            }

            // Case 1: one vector store sets all elements in a row to zero
            if (!needs_col_loop) {
                if (zeh != cfg.vlen) {
                    load_constant(col_loop_it, zeh);
                    vsetvli(x0, col_loop_it, vsew(src_sew) | vlmul(1));
                    // If stores use a greater vlen w.r.t the compute segment,
                    // the output register tail must be zero-initialized
                    if (zeh > cfg.vlen)
                        vxor_vv(vz, vz, vz);
                }

                L("loop_rows");
                add_constant(out, out, h_stride);
                vs(vz, out, src_sew);
            // Case 2: more than one vector store is required to zero elements
            } else {
                const gpr_t out_ptr = pool.pick(); // Row offset pointer
                const gpr_t vlen = pool.pick();    // This iteration vlen

                L("loop_rows");
                load_constant(col_loop_it, zeh);
                add_constant(out, out, h_stride);
                mv(out_ptr, out);

                L("loop_cols");
                vsetvli(vlen, col_loop_it, vsew(src_sew) | vlmul(1));
                vs(vz, out_ptr, src_sew);
                sub(col_loop_it, col_loop_it, vlen);
                slli(vlen, vlen, log2(src_sew));
                add(out_ptr, out_ptr, vlen);
                bnez(col_loop_it, "loop_cols");
            }
            // Update the row loop iterator and evaluate the branch
            if (needs_row_loop) {
                addi(row_loop_it, row_loop_it, -1);
                bnez(row_loop_it, "loop_rows");
            }
        }
        // If the vector length changed during this step, revert it back
        if (zeh != cfg.vlen) {
            ld(pool.head(), a0, offsetof(jit_conv_kernel_args_t, vlen));
            vsetvli(x0, pool.head(), vsew(src_sew) | vlmul(1));
        }
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
        if (is_bwdd)
            bwdd_strided_zero_elems(tmp_pool);
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
        case prop_kind::backward_data: {
            bwdd_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        case prop_kind::backward_weights: {
            bwdw_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        default:
            assert(!"unsupported propagation kind");
    }
        
    // Store Partial Sums Segment
    // TODO: this is temporary BIAS support
    if (cfg.with_bias && is_fwdd) {
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
    std::cout << "jit_convolution_kernel_t::fwdd_inner_loops" << std::endl;
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

void jit_convolution_kernel_t::bwdd_inner_loops(rvjit::vr_t *vout, int rb_sz, register_pool_t &rp) {
    auto const xcb = cfg.ocb;
    auto const wxcb = cfg.w_ocb;
    auto const kxcb_loop_sz = nstl::max(1, cfg.k_c/xcb);
    auto const xcb_loop_sz  = (cfg.k_c > xcb ? xcb : cfg.k_c) / wxcb;
    const int wei_sew = types::data_type_size(cfg.wei_dt);
    const int dst_sew = types::data_type_size(cfg.dst_dt);

    // ---------------------------- Tensor Offsets ----------------------------
    const int kw_off   = cfg.w_icb * cfg.w_ocb * wei_sew;
    const int xw_off   = xcb                   * dst_sew;
    const int wxcb_off = wxcb                  * dst_sew;
    int xcb_off        = cfg.oh * cfg.ow * xcb * dst_sew;

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
    ld(xsrc, a0, offsetof(jit_conv_kernel_args_t, dst));
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
            /// @details the calculations below are different in bwdd and fwdd
            /// The first vout index this loop after padding mask
            const int rw_s = traits.rbpadL > kw ? traits.rbpadL - kw : 0;
            /// The last vout index this loop after padding mask
            const int rw_e = traits.erbw - ((traits.rbpadR >= cfg.k_w - kw)
                ? (traits.rbpadR - (cfg.k_w - (kw+1)))
                : 0
            );

            if (rw_e - rw_s < 1) {
                ++wei_skip;
                continue;
            } else if (wei_skip) {
                /// @details the minus is unique to bwdd due to reverse iter
                // Adjust the weight ptr to account for padding
                load_constant(tmp, wei_skip * kw_off);
                sub(vsrc, vsrc, tmp);
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
            }

            // Loop end: one iteration over the non-vectorized dim block
            addi(xc_off, xc_off, dst_sew);
            bne(xc_off, xc_off_max, local_label);
            /// @details this is unique to bwdd due to reverse iterator
            wei_skip = 2;
        }
        wei_skip += cfg.kw - cfg.k_w;
    }
    wei_skip += cfg.kw * traits.rbpadB;
    /// @details this is unique to bwdd due to reverse iterator
    wei_skip = cfg.kh * cfg.kw + cfg.k_h * cfg.k_w - wei_skip;

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

void jit_convolution_kernel_t::bwdw_inner_loops(rvjit::vr_t *vout,
                                            int nvregs, register_pool_t &rp) {
    // Single-element width on tensor operands
    const int src_sew = types::data_type_size(cfg.src_dt);
    const int dst_sew = types::data_type_size(cfg.dst_dt);
    
    // Offset to the fields in the argument structure
    const int args_dst_off = offsetof(jit_conv_kernel_args_t, dst);
    const int args_src_off = offsetof(jit_conv_kernel_args_t, src);
    const int args_hloop_off = offsetof(jit_conv_kernel_args_t, h_loop_size);
    const int args_wloop_off = offsetof(jit_conv_kernel_args_t, w_loop_size);
    
    // Registers
    gpr_t vsrc_ptr = rp.pick();  // Tensor operand base address
    gpr_t xsrc_ptr = rp.pick();  // Scalar operand base address
    gpr_t xsrc_row_ptr = rp.pick();
    gpr_t vsrc_row_ptr = rp.pick();
    gpr_t h_iter = rp.pick();    // H dim loop iterator
    gpr_t w_iter = rp.pick();    // W dim loop iterator
    // FMA operand registers
    vr_t vsrc = static_cast<vr_t>(nvregs);
    fpr_t xsrc[4] = {ft0, ft1, ft2, ft3};
    static constexpr unsigned int nxsrc = 4;
    unsigned int xsrc_id = 0;

    // Spatial offsets between convolution windows
    int const dst_w_off = cfg.ocb * dst_sew;
    int const dst_h_off = cfg.ow * cfg.ocb * dst_sew;
    int const src_w_off = cfg.stride_w * cfg.icb * src_sew;
    int const src_h_off = cfg.stride_h * cfg.iw * cfg.icb * src_sew;
    
    // Assembly constats for the strides between convolution windows
    auto vw_off = asm_const(rp, cfg.vdim_is_oc ? dst_w_off : src_w_off);
    auto xw_off = asm_const(rp, cfg.vdim_is_oc ? src_w_off : dst_w_off);
    auto vh_off = asm_const(rp, cfg.vdim_is_oc ? dst_h_off : src_h_off);
    auto xh_off = asm_const(rp, cfg.vdim_is_oc ? src_h_off : dst_h_off);

    // Convolution Assembly
    ld(vsrc_ptr, a0, cfg.vdim_is_oc ? args_dst_off : args_src_off);
    prepare_constant(vh_off);
    prepare_constant(vw_off);
    ld(xsrc_ptr, a0, cfg.vdim_is_oc ? args_src_off : args_dst_off);
    prepare_constant(xh_off);
    prepare_constant(xw_off);
    ld(h_iter, a0, args_hloop_off);
    L("h_loop");
    ld(w_iter, a0, args_wloop_off);
    mv(vsrc_row_ptr, vsrc_ptr);
    mv(xsrc_row_ptr, xsrc_ptr);
    L("w_loop");
    vl(vsrc, vsrc_ptr, src_sew);
    for (int c = 0; c < cfg.rbc; ++c) {
        flw(xsrc[xsrc_id], xsrc_ptr, c * src_sew);
        vfmacc_vf(vout[c], xsrc[xsrc_id], vsrc);
        xsrc_id = (xsrc_id + 1) % nxsrc;
    }
    add_constant(vsrc_ptr, vsrc_ptr, vw_off);
    add_constant(xsrc_ptr, xsrc_ptr, xw_off);
    addi(w_iter, w_iter, -1);
    bnez(w_iter, "w_loop");
    add_constant(vsrc_ptr, vsrc_row_ptr, vh_off);
    add_constant(xsrc_ptr, xsrc_row_ptr, xh_off);
    addi(h_iter, h_iter, -1);
    bnez(h_iter, "h_loop");
}

} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv