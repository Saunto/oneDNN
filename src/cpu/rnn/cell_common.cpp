/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 * Common for RNN and LSTM cell execution
 */
#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/rnn/ref_rnn.hpp"
#include "cpu/simple_q10n.hpp"
#if DNNL_X64
#include "cpu/x64/rnn/brgemm_cell_common.hpp"
#endif
#include <functional>
#include "cpu/x64/rnn/brgemm_cell_common_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
using namespace rnn_utils;
using namespace dnnl::impl::utils;
#if DNNL_X64
using namespace dnnl::impl::cpu::x64;
#endif
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution_ref)) {
    auto weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;
    auto weights_projection_scales = rnn.is_lstm_projection
            ? pd_->attr()->rnn_weights_projection_qparams_.scales_
            : nullptr;

    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    if (rnn.need_gemm_layer(cell_position)) {
        CHECK((this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
                rnn.slc, 1.0f, w_layer_[0], rnn.weights_layer_ld, src_layer_,
                src_layer_ld, 0.0f, scratch_gates_, rnn.scratch_gates_ld));
    }
    CHECK((this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
            rnn.sic, 1.0f, w_iter_[0], rnn.weights_iter_ld, src_iter_,
            src_iter_ld, 1.0f, scratch_gates_, rnn.scratch_gates_ld));

    // Note: here proj_ht is scratchpad if inference or workspace if training
    auto dst_postgemm = rnn.is_lstm_projection ? proj_ht_ : dst_layer_;
    // for lstmp, the copy to dst_iter happens after the projection
    auto dst_iter_postgemm = rnn.is_lstm_projection ? nullptr : dst_iter_;
    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_postgemm, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, weights_peephole_, bias_[0], ws_grid_,
            scratch_cell_, dst_iter_postgemm, weights_scales,
            rnn.dhc * sizeof(scratch_t));

    if (rnn.is_lstm_projection) {
        auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);

        // Here, because the accumulation type is different
        // than dst_layer, we have to use scratch to hold temporary
        // accumulators
        assert(rnn.scratch_gates_ld >= rnn.dlc);
        gemm_acc_t *dst_proj = rnn.dt_conf == all_f32 ? (gemm_acc_t *)dst_layer_
                                                      : scratch_gates_;
        int dst_proj_ld
                = rnn.dt_conf == all_f32 ? dst_layer_ld : rnn.scratch_gates_ld;

        CHECK((this->*gemm_projection_func)('N', 'N', rnn.dic, rnn.mb, rnn.dhc,
                1.0f, w_projection_[0], rnn.weights_projection_ld, dst_postgemm,
                rnn.proj_ht_ld, 0.0f, dst_proj, dst_proj_ld));

        // we have to downconvert the output to dst_layer_t and copy to dst_iter if needed
        rnn_postgemm_->execute_part2(rnn, cell_position, nullptr, dst_proj,
                dst_layer_, nullptr, nullptr, w_proj_comp, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, dst_iter_, weights_projection_scales,
                rnn.dlc * sizeof(dst_layer_t));
    }

    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_ref);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_ref);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_ref);
template rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_ref);

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution_brgemm_fwd)) {
#if DNNL_X64
    auto weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;
    auto weights_projectons_scales = rnn.is_lstm_projection
            ? pd_->attr()->rnn_weights_projection_qparams_.scales_
            : nullptr;
    // pased the const weight scale to postgemm to avoid the shift
    // in scale array for per_oc in brgemm
    const auto wscales_postgemm = pd_->attr()->rnn_weights_qparams_.scales_;
    const auto wscales_proj_postgemm = rnn.is_lstm_projection
            ? pd_->attr()->rnn_weights_projection_qparams_.scales_
            : nullptr;

    auto dst_postgemm = rnn.is_lstm_projection ? proj_ht_ : dst_layer_;
    auto dst_iter_postgemm = rnn.is_lstm_projection ? nullptr : dst_iter_;

    const dim_t layer_desc_idx = rnn.layer_brgemm_desc(cell_position);
    const dim_t iter_desc_idx = rnn.iter_brgemm_desc(cell_position);

    const auto LDAl = rnn.src_layer_ld(cell_position);
    const auto LDAi = rnn.src_iter_ld(cell_position);
    const auto LDAic = rnn.src_iter_c_ld(cell_position);

    const auto LDDl = rnn.dst_layer_ld(cell_position);
    const auto LDDi = rnn.dst_iter_ld(cell_position);
    const auto LDDic = rnn.dst_iter_c_ld(cell_position);

    auto Al = src_layer_;
    auto Ai = src_iter_;
    auto Aic = src_iter_c_;

    auto Dpg = dst_postgemm;
    auto Di = dst_iter_postgemm;
    auto Dic = dst_iter_c_;

    auto Bl = w_layer_[0];
    auto Bi = w_iter_[0];
    auto C = scratch_gates_;

    const int Bl_n_offset = rnn.K1padded * rnn.n_block;
    const int Bi_n_offset = rnn.K2padded * rnn.n_block;
    const int Bl_g_offset = rnn.N_blocks * Bl_n_offset;
    const int Bi_g_offset = rnn.N_blocks * Bi_n_offset;
    const int Al_k_tail_offset = rnn.KB1_blocks * rnn.k1_block;
    const int Ai_k_tail_offset = rnn.KB2_blocks * rnn.k2_block;
    const int Bl_kb_offset = rnn.k1_block * rnn.n_block;
    const int Bi_kb_offset = rnn.k2_block * rnn.n_block;
    const int Bl_k_tail_offset = rnn.KB1_blocks * rnn.k1_block * rnn.n_block;
    const int Bi_k_tail_offset = rnn.KB2_blocks * rnn.k2_block * rnn.n_block;

    const int Nblocking = (rnn.unfused_post_gemm) ? rnn.N_blocks * rnn.n_gates
                                                  : rnn.N_blocks;
    const int n_gates = (rnn.unfused_post_gemm) ? 1 : rnn.n_gates;

    const int max_nthr = rnn.nthr;
    const int work_amount = Nblocking * rnn.M_blocks;

    const int mask = pd_->attr()->rnn_weights_qparams_.mask_;
    const int pmask = rnn.is_lstm_projection
            ? pd_->attr()->rnn_weights_projection_qparams_.mask_
            : 0;

    auto brgemm_kernel_iter_n_tail = (rnn.need_gemm_layer(cell_position))
            ? rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx].get()
            : rnn_brgemm_.kernel_iter_N_tail_b0_[iter_desc_idx].get();
    auto brgemm_kernel_iter_main = (rnn.need_gemm_layer(cell_position))
            ? rnn_brgemm_.kernel_iter_b1_[iter_desc_idx].get()
            : rnn_brgemm_.kernel_iter_b0_[iter_desc_idx].get();

    const bool k_blocks_eq = rnn.k1_block == rnn.k2_block;
    const char *pallete_buff_iter_main = k_blocks_eq
            ? rnn_brgemm_.pallete_buff_layer_
            : rnn_brgemm_.pallete_buff_iter_;
    const char *pallete_buff_layer_n_tail = k_blocks_eq
            ? rnn_brgemm_.pallete_buff_layer_n_tail_
            : rnn_brgemm_.pallete_buff_iter_n_tail_;

    parallel(max_nthr, [&](const int ithr, const int nthr) {
        gemm_acc_t *amx_buffer = nullptr;
        x64::brgemm_batch_element_t *addr_batch = nullptr;

        int start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);

        const int max_K_Block = nstl::max(rnn.KB1_blocks + 1,
                nstl::max(rnn.KBproj_blocks + 1, rnn.KB2_blocks + 1));
        addr_batch = addr_batch_global + ithr * max_K_Block;

        if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
            amx_buffer = amx_scratchpad + rnn.m_block * rnn.n_block * ithr;
        }

        int nb_i = 0, mb = 0;
        nd_iterator_init(start, nb_i, Nblocking, mb, rnn.M_blocks);

        amx_tile_configuration_loader_t load_cfg_if_needed;

        while (start < end) {
            const int m = mb * rnn.m_block;

            const int nb = (rnn.unfused_post_gemm) ? nb_i / rnn.n_gates : nb_i;
            const int n = nb * rnn.n_block;

            const int g_unfused
                    = (rnn.unfused_post_gemm) ? nb_i % rnn.n_gates : 0;

            auto Al_m = Al + m * LDAl;
            auto Ai_m = Ai + m * LDAi;
            auto Bl_n = Bl + nb * Bl_n_offset;
            auto Bi_n = Bi + nb * Bi_n_offset;
            auto C_n = C + m * rnn.LDC + n;

            auto Aic_n = Aic + m * LDAic + n;
            auto Dpg_n = (Dpg != nullptr) ? Dpg + m * LDDl + n : nullptr;
            auto Di_n = (Di != nullptr) ? Di + m * LDDi + n : nullptr;
            auto Dic_n = (Dic != nullptr) ? Dic + m * LDDic + n : nullptr;
            auto bias_n = bias_[0] + n;
            const float *weights_peephole_n = weights_peephole_ + n;
            auto weights_scales_n = weights_scales + ((mask) ? n : 0);

            const bool do_n_tail = (n + rnn.n_block) > rnn.N;
            int block_step = 0;
            brgemm_kernel_t *brgemm_kernel_layer_b0;
            brgemm_kernel_t *brgemm_kernel_iter;
            brgemm_kernel_t *brgemm_kernel_layer_tail;
            brgemm_kernel_t *brgemm_kernel_iter_tail;

            if (do_n_tail) {
                block_step = rnn.n_tail * sizeof(scratch_t);
                brgemm_kernel_layer_b0
                        = rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx]
                                  .get();
                brgemm_kernel_iter = brgemm_kernel_iter_n_tail;
                brgemm_kernel_layer_tail
                        = rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx]
                                  .get();
                brgemm_kernel_iter_tail
                        = rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx]
                                  .get();

            } else {
                block_step = rnn.n_block * sizeof(scratch_t);
                brgemm_kernel_layer_b0
                        = rnn_brgemm_.kernel_layer_b0_[layer_desc_idx].get();
                brgemm_kernel_iter = brgemm_kernel_iter_main;
                brgemm_kernel_layer_tail
                        = rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx]
                                  .get();
                brgemm_kernel_iter_tail
                        = rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx]
                                  .get();
            }
            if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {

                const char *pallete_buff_iter = pallete_buff_iter_main;
                const char *pallete_buff_layer
                        = rnn_brgemm_.pallete_buff_layer_;

                if (do_n_tail) {
                    pallete_buff_iter = pallete_buff_layer_n_tail;
                    pallete_buff_layer = rnn_brgemm_.pallete_buff_layer_n_tail_;
                }

                for (int g = 0; g < n_gates; g++) {
                    const int lg = g + g_unfused;
                    auto Bl_g = Bl_n + lg * Bl_g_offset;
                    auto Bi_g = Bi_n + lg * Bi_g_offset;
                    auto C_g = C_n + lg * rnn.N;

                    if (rnn.need_gemm_layer(cell_position)) {
                        for (int k1 = 0; k1 < rnn.KB1_blocks; k1++) {
                            addr_batch[k1].ptr.A = Al_m + k1 * rnn.k1_block;
                            addr_batch[k1].ptr.B = Bl_g + k1 * Bl_kb_offset;
                        }
                        load_cfg_if_needed(pallete_buff_layer);
                        brgemm_kernel_execute(brgemm_kernel_layer_b0,
                                rnn.KB1_blocks, addr_batch, (void *)C_g,
                                amx_buffer);
                    }

                    for (int k2 = 0; k2 < rnn.KB2_blocks; k2++) {
                        addr_batch[k2].ptr.A = Ai_m + k2 * rnn.k2_block;
                        addr_batch[k2].ptr.B = Bi_g + k2 * Bi_kb_offset;
                    }

                    load_cfg_if_needed(pallete_buff_iter);
                    brgemm_kernel_execute(brgemm_kernel_iter, rnn.KB2_blocks,
                            addr_batch, (void *)C_g, amx_buffer);
                }

                if (rnn.k1_tail || rnn.k2_tail) {
                    const char *tail_cfg_k1, *tail_cfg_k2;
                    if (do_n_tail) {
                        tail_cfg_k1 = rnn_brgemm_.pallete_buff_nk1_tail_;
                        tail_cfg_k2 = rnn_brgemm_.pallete_buff_nk2_tail_;
                    } else {
                        tail_cfg_k1 = rnn_brgemm_.pallete_buff_k1_tail_;
                        tail_cfg_k2 = rnn_brgemm_.pallete_buff_k2_tail_;
                    }
                    if (rnn.k1_tail && rnn.need_gemm_layer(cell_position)) {
                        load_cfg_if_needed(tail_cfg_k1);
                        for (int g = 0; g < n_gates; g++) {
                            int lg = g + g_unfused;
                            auto Bl_g = Bl_n + lg * Bl_g_offset;
                            auto C_g = C_n + lg * rnn.N;

                            addr_batch[0].ptr.A = Al_m + Al_k_tail_offset;
                            addr_batch[0].ptr.B = Bl_g + Bl_k_tail_offset;
                            brgemm_kernel_execute(brgemm_kernel_layer_tail, 1,
                                    addr_batch, (void *)C_g, amx_buffer);
                        }
                    }
                    if (rnn.k2_tail) {
                        load_cfg_if_needed(tail_cfg_k2);
                        for (int g = 0; g < n_gates; g++) {
                            int lg = g + g_unfused;
                            auto Bi_g = Bi_n + lg * Bi_g_offset;
                            auto C_g = C_n + lg * rnn.N;

                            addr_batch[0].ptr.A = Ai_m + Ai_k_tail_offset;
                            addr_batch[0].ptr.B = Bi_g + Bi_k_tail_offset;
                            brgemm_kernel_execute(brgemm_kernel_iter_tail, 1,
                                    addr_batch, (void *)C_g, amx_buffer);
                        }
                    }
                }
            } else {
                for (int g = 0; g < n_gates; g++) {
                    const int lg = g + g_unfused;
                    const auto Bl_g = Bl_n + lg * Bl_g_offset;
                    const auto Bi_g = Bi_n + lg * Bi_g_offset;
                    const auto C_g = C_n + lg * rnn.N;

                    if (rnn.need_gemm_layer(cell_position)) {
                        for (int i = 0; i < rnn.KB1_blocks; i++) {
                            addr_batch[i].ptr.A = Al_m + i * rnn.k1_block;
                            addr_batch[i].ptr.B = Bl_g + i * Bl_kb_offset;
                        }
                        brgemm_kernel_execute(brgemm_kernel_layer_b0,
                                rnn.KB1_blocks, addr_batch, (void *)C_g,
                                amx_buffer);

                        if (rnn.k1_tail) {
                            addr_batch[0].ptr.A = Al_m + Al_k_tail_offset;
                            addr_batch[0].ptr.B = Bl_g + Bl_k_tail_offset;
                            brgemm_kernel_execute(brgemm_kernel_layer_tail, 1,
                                    addr_batch, (void *)C_g, amx_buffer);
                        }
                    }

                    for (int i = 0; i < rnn.KB2_blocks; i++) {
                        addr_batch[i].ptr.A = Ai_m + i * rnn.k2_block;
                        addr_batch[i].ptr.B = Bi_g + i * Bi_kb_offset;
                    }
                    brgemm_kernel_execute(brgemm_kernel_iter, rnn.KB2_blocks,
                            addr_batch, (void *)C_g, amx_buffer);

                    if (rnn.k2_tail) {
                        addr_batch[0].ptr.A = Ai_m + Ai_k_tail_offset;
                        addr_batch[0].ptr.B = Bi_g + Bi_k_tail_offset;
                        brgemm_kernel_execute(brgemm_kernel_iter_tail, 1,
                                addr_batch, (void *)C_g, amx_buffer);
                    }
                }
            }
            if (!rnn.unfused_post_gemm) {
                const auto curr_ws_gates_ = ws_gates_ + (m * rnn.ws_gates_ld)
                        + nb_i * rnn.n_block;
                rnn_postgemm_->execute(rnn, cell_position, curr_ws_gates_, C_n,
                        Dpg_n, Dic_n, Ai_m, Aic_n, diff_src_layer_,
                        diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                        diff_dst_iter_, diff_dst_iter_c_, weights_peephole_n,
                        bias_n, ws_grid_, scratch_cell_, Di_n, weights_scales_n,
                        block_step);
            }
            ++start;
            nd_iterator_step(nb_i, Nblocking, mb, rnn.M_blocks);
        }
    });
    if (rnn.unfused_post_gemm) {
        rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
                dst_postgemm, dst_iter_c_, src_iter_, src_iter_c_,
                diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                weights_peephole_, bias_[0], ws_grid_, scratch_cell_,
                dst_iter_postgemm, wscales_postgemm,
                rnn.dhc * sizeof(scratch_t));
    }
    if (rnn.is_lstm_projection) {
        // Here, because the accumulation type is likely different
        // than dst_iter, we have to use scratch to hold temporary
        // accumulators
        // TODO: for projection, directly use a type(dst_iter) for output
        auto dst_layer_ld = rnn.dst_layer_ld(cell_position, true);
        int proj_desc_idx = (rnn.dt_conf == all_f32)
                ? rnn.dst_brgemm_desc(cell_position, true)
                : 0;

        int work_amount_proj = rnn.Nproj_blocks * rnn.M_blocks;

        auto Ap = dst_postgemm;
        auto Bp = w_projection_[0];
        gemm_acc_t *Cp = (rnn.dt_conf == all_f32) ? (gemm_acc_t *)dst_layer_
                                                  : scratch_gates_;
        auto Di = dst_iter_;
        auto Dl = dst_layer_;
        auto Wp_comp = w_proj_comp;

        const dim_t LDC
                = rnn.dt_conf == all_f32 ? dst_layer_ld : rnn.scratch_gates_ld;
        const int pLDDl = dst_layer_ld;

        const dim_t B_n_offset = rnn.Kprojpadded * rnn.n_block;
        const dim_t Bp_kb_offset = rnn.kproj_block * rnn.n_block;

        parallel(max_nthr, [&](const int ithr, const int nthr) {
            int start = 0, end = 0;
            balance211(work_amount_proj, nthr, ithr, start, end);
            gemm_acc_t *amx_buffer = nullptr;
            brgemm_batch_element_t *addr_batch = nullptr;
            amx_tile_configuration_loader_t load_cfg_if_needed;

            if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                int max_K_Block = nstl::max(rnn.KB1_blocks + 1,
                        nstl::max(rnn.KBproj_blocks + 1, rnn.KB2_blocks + 1));
                addr_batch = addr_batch_global + ithr * max_K_Block;

                amx_buffer = amx_scratchpad + rnn.m_block * rnn.n_block * ithr;
                load_cfg_if_needed(rnn_brgemm_.pallete_buff_proj_);
            } else {
                addr_batch = addr_batch_global + ithr;
            }

            int nb = 0, mb = 0;
            nd_iterator_init(start, nb, rnn.Nproj_blocks, mb, rnn.M_blocks);
            while (start < end) {
                int n = nb * rnn.n_block;
                int m = mb * rnn.m_block;
                bool do_n_tail = (n + rnn.n_block) > rnn.Nproj;

                int block_step = ((do_n_tail) ? rnn.nproj_tail : rnn.n_block)
                        * sizeof(dst_layer_t);
                auto weights_scales_n
                        = weights_projectons_scales + ((pmask) ? n : 0);

                auto Ap_m = Ap + m * rnn.LDAproj;
                auto Bp_n = Bp + nb * B_n_offset;
                auto Cp_n = Cp + m * LDC + n;
                auto Di_n = (Di != nullptr) ? Di + m * LDDi + n : nullptr;
                auto Dl_n = (Dl != nullptr) ? Dl + m * pLDDl + n : nullptr;
                auto Wp_comp_n = Wp_comp + n;

                brgemm_kernel_t *brgemm_kernel_proj_b0 = (do_n_tail)
                        ? rnn_brgemm_.kernel_proj_N_tail_b0_[proj_desc_idx]
                                  .get()
                        : rnn_brgemm_.kernel_proj_b0_[proj_desc_idx].get();

                if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                    if (do_n_tail)
                        load_cfg_if_needed(
                                rnn_brgemm_.pallete_buff_nproj_tail_);
                    for (int k = 0; k < rnn.KBproj_blocks; k++) {
                        addr_batch[k].ptr.A = Ap_m + k * rnn.kproj_block;
                        addr_batch[k].ptr.B = Bp_n + k * Bp_kb_offset;
                    }
                    brgemm_kernel_execute(brgemm_kernel_proj_b0,
                            rnn.KBproj_blocks, addr_batch, (void *)Cp_n,
                            amx_buffer);
                    if (rnn.kproj_tail) {
                        brgemm_kernel_t *brgemm_kernel_proj_tail;
                        const char *tail_cfg_kproj, *tail_recfg;
                        if (do_n_tail) {
                            tail_cfg_kproj
                                    = rnn_brgemm_.pallete_buff_nkproj_tail_;
                            tail_recfg = rnn_brgemm_.pallete_buff_nproj_tail_;
                            brgemm_kernel_proj_tail
                                    = rnn_brgemm_
                                              .kernel_proj_NK_tail_b1_
                                                      [proj_desc_idx]
                                              .get();
                        } else {
                            tail_cfg_kproj
                                    = rnn_brgemm_.pallete_buff_kproj_tail_;
                            tail_recfg = rnn_brgemm_.pallete_buff_proj_;
                            brgemm_kernel_proj_tail
                                    = rnn_brgemm_
                                              .kernel_proj_K_tail_b1_
                                                      [proj_desc_idx]
                                              .get();
                        }
                        load_cfg_if_needed(tail_cfg_kproj);
                        addr_batch[0].ptr.A
                                = Ap_m + rnn.KBproj_blocks * rnn.kproj_block;
                        addr_batch[0].ptr.B = Bp_n
                                + rnn.KBproj_blocks * rnn.kproj_block
                                        * rnn.n_block;
                        brgemm_kernel_execute(brgemm_kernel_proj_tail, 1,
                                addr_batch, (void *)Cp_n, amx_buffer);
                        load_cfg_if_needed(tail_recfg);
                    }
                } else {
                    addr_batch[0].ptr.A = Ap_m;
                    addr_batch[0].ptr.B = Bp_n;
                    brgemm_kernel_execute(brgemm_kernel_proj_b0, 1, addr_batch,
                            (void *)Cp_n, amx_buffer);
                }
                if (!rnn.unfused_post_gemm) {
                    rnn_postgemm_->execute_part2(rnn, cell_position, nullptr,
                            Cp_n, Dl_n, nullptr, nullptr, Wp_comp_n, nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, Di_n,
                            weights_scales_n, block_step);
                }
                ++start;
                nd_iterator_step(nb, rnn.Nproj_blocks, mb, rnn.M_blocks);
            }
        });
        // we have to downconvert the output to dst_layer_t and copy to dst_iter if needed
        if (rnn.unfused_post_gemm) {
            rnn_postgemm_->execute_part2(rnn, cell_position, nullptr, Cp,
                    dst_layer_, nullptr, nullptr, w_proj_comp, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, dst_iter_, wscales_proj_postgemm,
                    rnn.dlc * sizeof(dst_layer_t));
        }
    }
#endif
    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_brgemm_fwd);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_brgemm_fwd);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm_fwd);
template rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm_fwd);

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm_fwd) {
    return dnnl_success;
}
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_brgemm_fwd) {
    return dnnl_success;
}

template <typename scratch_data_t, typename acc_data_t>
void lstm_bwd_weights_peephole_and_bias(const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, const float *src_iter_c_,
        const float *dst_iter_c_, const scratch_data_t *scratch_gates_,
        float *diff_weights_peephole_, acc_data_t *diff_bias_) {
    auto dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    auto src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

    ws_states_iter_c_aoc<const float> dst_iter_c(
            rnn, dst_iter_c_, dst_iter_c_ld);
    ws_states_iter_c_aoc<const float> src_iter_c(
            rnn, src_iter_c_, src_iter_c_ld);
    ws_gates_aoc<const scratch_data_t> scratch_gates(rnn, scratch_gates_);
    weights_peephole_aoc_t<float> diff_weights_peephole(
            rnn, diff_weights_peephole_);

    parallel(0, [&](int ithr, int nthr) {
        int g_dhc_start {}, g_dhc_stop {};
        const int gates_to_process = 5; // 3 -- weights peephole +
                // 2 -- bias (process a pair at once)
        balance211(gates_to_process * rnn.dhc, nthr, ithr, g_dhc_start,
                g_dhc_stop);
        int g = g_dhc_start / rnn.dhc;
        int dhc = g_dhc_start % rnn.dhc;
        while (g_dhc_start++ < g_dhc_stop) {
            if (g < 3) {
                // weights peephole
                auto &c_states = g < 2 ? src_iter_c : dst_iter_c;
                const int scratch_g = g < 2 ? g : 3;
                for (int mb = 0; mb < rnn.mb; ++mb) {
                    diff_weights_peephole(g, dhc) += c_states(mb, dhc)
                            * scratch_gates(mb, scratch_g, dhc);
                }
            } else {
                // bias
                const int bias_g_start = 2 * (g - 3);
                const int bias_g_end = bias_g_start + 2;
                for_(int bias_g = bias_g_start; bias_g < bias_g_end; ++bias_g)
                for (int mb = 0; mb < rnn.mb; ++mb)
                    diff_bias_[bias_g * rnn.dhc + dhc]
                            += scratch_gates(mb, bias_g, dhc);
            }
            if (++dhc == rnn.dhc) {
                dhc = 0;
                g++;
            }
        }
    });
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename weights_data_t, typename src_data_t,
        typename acc_data_t, typename scratch_data_t>
dnnl_status_t common_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_proj_f, T4 gemm_weights_layer_f, T5 gemm_weights_iter_f,
        T6 gemm_weights_proj_f, T7 rnn_postgemm,
        const rnn_utils::rnn_conf_t &rnn, const cell_position_t cell_position,
        src_data_t *dst_layer_, float *dst_iter_c_, acc_data_t *diff_src_layer_,
        acc_data_t *diff_src_iter_, acc_data_t *diff_src_iter_c_,
        weights_data_t **w_layer_, weights_data_t **w_iter_,
        weights_data_t **w_proj_, const float *weights_peephole_, float **bias_,
        const src_data_t *src_layer_, const src_data_t *src_iter_,
        const float *src_iter_c_, acc_data_t *diff_dst_layer_,
        acc_data_t *diff_dst_iter_, acc_data_t *diff_dst_iter_c_,
        acc_data_t *diff_w_layer_, acc_data_t *diff_w_iter_,
        float *diff_weights_projection_, float *diff_weights_peephole_,
        acc_data_t *diff_bias_, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *ws_ht_,
        acc_data_t *scratch_diff_ht_, src_data_t *ws_grid_,
        scratch_data_t *scratch_cell_, src_data_t *dst_iter_) {

    if (rnn.is_lstm_projection) {
        parallel_nd(rnn.mb, [&](int i) {
            PRAGMA_OMP_SIMD()
            for (int j = 0; j < rnn.dlc; j++)
                scratch_diff_ht_[i * rnn.scratch_diff_ht_ld + j]
                        = diff_dst_layer_[i * rnn.ws_diff_states_layer_ld + j]
                        + diff_dst_iter_[i * rnn.ws_diff_states_iter_ld + j];
        });

        CHECK(gemm_weights_proj_f(
                scratch_diff_ht_, ws_ht_, diff_weights_projection_));
        CHECK(gemm_proj_f(w_proj_[0], scratch_diff_ht_, diff_dst_layer_));
    }

    rnn_postgemm->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, weights_peephole_, bias_[0], ws_grid_,
            scratch_cell_, dst_iter_, nullptr, 0);

    /// bwd by data on the cell
    CHECK(gemm_iter_f(w_iter_[0], scratch_gates_, diff_src_iter_));

    /// bwd by weights on the cell
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(scratch_gates_, src_layer_, diff_w_layer_));

    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    if (!rnn.merge_gemm_iter)
        CHECK(gemm_weights_iter_f(scratch_gates_, src_iter_, diff_w_iter_));

    if (rnn.is_lstm_peephole) {
        /// bwd by weights peephole and bias
        lstm_bwd_weights_peephole_and_bias(rnn, cell_position, src_iter_c_,
                dst_iter_c_, scratch_gates_, diff_weights_peephole_,
                diff_bias_);
    } else {
        /// bwd by bias we just accumulate diffs from the gates
        gates_reduction(rnn, scratch_gates_, diff_bias_);
    }
    return dnnl_success;
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_ref) {
    auto gemm_layer = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_iter_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_proj = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_projection_func)('N', 'N', rnn.dhc, rnn.mb, rnn.dic,
                1.0, A, rnn.weights_projection_ld, B, rnn.scratch_diff_ht_ld,
                0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_weights_layer = [&](const float *A, const float *B, float *C) {
        auto src_layer_ld = rnn.src_layer_ld(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, src_layer_ld, 1.0, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter = [&](const float *A, const float *B, float *C) {
        auto src_iter_ld = rnn.src_iter_ld(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, src_iter_ld, 1.0, C,
                rnn.diff_weights_iter_ld);
    };
    auto gemm_weights_proj = [&](const float *A, const float *B, float *C) {
        return gemm('N', 'T', rnn.dlc, rnn.dhc, rnn.mb, 1.0f, A,
                rnn.scratch_diff_ht_ld, B, rnn.ws_ht_ld, 1.0f, C,
                rnn.diff_weights_projection_ld);
    };
    return common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_proj,
            gemm_weights_layer, gemm_weights_iter, gemm_weights_proj,
            rnn_postgemm_, rnn, cell_position, dst_layer_, dst_iter_c_,
            diff_src_layer_, diff_src_iter_, diff_src_iter_c_, w_layer_,
            w_iter_, w_projection_, weights_peephole_, bias_, src_layer_,
            src_iter_, src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, diff_w_layer_, diff_w_iter_,
            diff_weights_projection_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, proj_ht_, scratch_diff_ht_, ws_grid_,
            scratch_cell_, dst_iter_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_ref) {
    auto gemm_layer = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_iter_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_proj = [&](const bfloat16_t *, const float *, float *) {
        assert(!"unimplemented");
        return dnnl_unimplemented;
    };
    auto gemm_weights_layer
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_layer_ld = rnn.src_layer_ld(cell_position);
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb,
                          1.0, A, rnn.scratch_gates_ld, B, src_layer_ld, 1.0, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_iter_ld = rnn.src_iter_ld(cell_position);
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb,
                          1.0, A, rnn.scratch_gates_ld, B, src_iter_ld, 1.0, C,
                          rnn.diff_weights_iter_ld);
              };
    auto gemm_weights_proj = [&](const float *, const bfloat16_t *, float *) {
        assert(!"unimplemented");
        return dnnl_unimplemented;
    };
    return common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_proj,
            gemm_weights_layer, gemm_weights_iter, gemm_weights_proj,
            rnn_postgemm_, rnn, cell_position, dst_layer_, dst_iter_c_,
            diff_src_layer_, diff_src_iter_, diff_src_iter_c_, w_layer_,
            w_iter_, w_projection_, weights_peephole_, bias_, src_layer_,
            src_iter_, src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, diff_w_layer_, diff_w_iter_,
            diff_weights_projection_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, proj_ht_, scratch_diff_ht_, ws_grid_,
            scratch_cell_, dst_iter_);
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution_brgemm_bwd)) {

    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, weights_peephole_, bias_[0], ws_grid_,
            scratch_cell_, dst_iter_, nullptr, 0);

#if DNNL_X64
    using brgemm_diff_src_calc_t = x64::brgemm_diff_src_layer_iter_t<weights_t,
            scratch_t, gemm_acc_t>;
    using brgemm_diff_weights_calc_t
            = x64::brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t,
                    scratch_t, gemm_acc_t>;

    const brgemm_diff_src_calc_t diff_src_calc(rnn_brgemm_, rnn, cell_position,
            scratch_gates_, w_iter_[0], w_layer_[0], diff_src_iter_,
            diff_src_layer_, amx_scratchpad, addr_batch_global);
    const brgemm_diff_weights_calc_t diff_weights_calc(rnn_brgemm_, rnn,
            cell_position, src_iter_, scratch_src_iter_, src_layer_,
            scratch_src_layer_, scratch_gates_, scratch_gates_blocked_,
            diff_w_iter_, diff_w_layer_, diff_bias_, amx_scratchpad,
            addr_batch_global);

    // calculate
    // dff_src_iter = scratch * w_iter
    // dff_src_layer = scratch * w_layer
    diff_src_calc.execute();

    if (rnn.diff_wei_brgemm.global_transpose) {
        const auto layer_transpose = src_layer_iter_transpose_t(
                rnn, rnn.slc, rnn.src_layer_ld(cell_position));
        const auto iter_transpose = src_layer_iter_transpose_t(
                rnn, rnn.sic, rnn.src_iter_ld(cell_position));
        layer_transpose.execute_in_parallel(src_layer_, scratch_src_layer_);
        iter_transpose.execute_in_parallel(src_iter_, scratch_src_iter_);
    }
    // calculate
    // dff_weights_layer = src_layer^T * scratch
    // dff_weights_iter = src_iter^T * scratch
    // performs gates reductions
    // diff_bias = scratch reduction over mb
    diff_weights_calc.execute();
#endif

    return dnnl_success;
}

template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_brgemm_bwd) {
    return dnnl_success;
}
template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_brgemm_bwd) {
    return dnnl_success;
}
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm_bwd) {
    return dnnl_success;
}
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm_bwd) {
    return dnnl_success;
}
template rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm_bwd);
template rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_brgemm_bwd);

} // namespace cpu
} // namespace impl
} // namespace dnnl
