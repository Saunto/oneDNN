/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP
#define CPU_X64_RNN_JIT_UNI_LSTM_CELL_POSTGEMM_FWD_HPP

#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_lstm_cell_postgemm_fwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lstm_cell_postgemm_fwd)

    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    jit_uni_lstm_cell_postgemm_fwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd) {}

    ~jit_uni_lstm_cell_postgemm_fwd() {
        delete sigmoid_injector_;
        delete tanh_injector_;
    }

    status_t init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        // we use rax for both constant tables and load correspondent label
        // into it when calling correspondent injector.
        sigmoid_injector_ = new injector_t(
                this, alg_kind::eltwise_logistic, 0.0f, 0.0f, 1.0f, true, rax);
        tanh_injector_ = new injector_t(
                this, alg_kind::eltwise_tanh, 0.0f, 0.0f, 1.0f, true, rax);
        return create_kernel();
    }

protected:
    injector_t *sigmoid_injector_;
    injector_t *tanh_injector_;

    // register size in bytes
    using Vmm = typename jit_uni_eltwise_injector_f32<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_dst
            = vlen / (sizeof(float) / types::data_type_size(src_data_t));
    size_t cstate_dt_size = sizeof(float);
    size_t hstate_dt_size = types::data_type_size(src_data_t);
    size_t gate_dt_size = types::data_type_size(src_data_t);
    size_t scratch_dt_size = types::data_type_size(scratch_data_t);
    size_t qscale_dt_size = sizeof(float);
    size_t weights_peephole_dt_size = sizeof(float);
    size_t bias_dt_size = sizeof(float);

    static constexpr int tmp_id_begin_ = 5;
    const int tmp_id_end_ = cpu_isa_traits<isa>::n_vregs
            - ((bf16_emu_ && is_superset(isa, avx512_common)) ? 4 : 0);
    int current_tmp_id_ = tmp_id_begin_;
    const bool avx2_available_ = is_superset(isa, avx2);

    Vmm get_next_tmp_vmm() {
        const Vmm vmm {current_tmp_id_++};

        if (current_tmp_id_ == tmp_id_end_) current_tmp_id_ = tmp_id_begin_;

        return vmm;
    }

    Xbyak::Xmm get_next_tmp_xmm() {
        return Xbyak::Xmm(get_next_tmp_vmm().getIdx());
    }

    void vaddps_rhs_op_mem(
            const Vmm &dst, const Vmm &lhs, const Xbyak::Address &rhs_addr) {

        if (avx2_available_)
            uni_vaddps(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_vmm();
            uni_vmovups(rhs, rhs_addr);
            uni_vaddps(dst, lhs, rhs);
        }
    }

    void vfmadd231ps_rhs_op_mem(
            const Vmm &dst, const Vmm &lhs, const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            uni_vfmadd231ps(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_vmm();
            uni_vmovups(rhs, rhs_addr);
            uni_vfmadd231ps(dst, lhs, rhs);
        }
    }

    void vaddss_rhs_op_mem(const Xbyak::Xmm &dst, const Xbyak::Xmm &lhs,
            const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            uni_vaddss(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_xmm();
            uni_vmovss(rhs, rhs_addr);
            uni_vaddss(dst, lhs, rhs);
        }
    }

    void vfmadd231ss_rhs_op_mem(const Xbyak::Xmm &dst, const Xbyak::Xmm &lhs,
            const Xbyak::Address &rhs_addr) {
        if (avx2_available_)
            uni_vfmadd231ss(dst, lhs, rhs_addr);
        else {
            const auto rhs = get_next_tmp_xmm();
            uni_vmovss(rhs, rhs_addr);
            uni_vfmadd231ss(dst, lhs, rhs);
        }
    }

    void generate() override {
        using namespace Xbyak;

        auto is_training
                = (pd_->desc()->prop_kind == prop_kind::forward_training);

        int mask = pd_->attr()->rnn_weights_qparams_.mask_;
        float *weights_scales = pd_->attr()->rnn_weights_qparams_.scales_;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;

        // Register map
        Reg64 loop_cnt(rbx); // loop counter
        // We skip vmm0 as it can be used by the injector for masks on sse4.1

        // We start code generations here
        preamble();

        Reg64 n_step_reg(rbp);

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_scratch_gates_reg = abi_param2;
        auto addr_weights_peephole_reg = r11;
        auto addr_bias_reg = abi_param3;
        auto addr_states_t_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_states_t_l_copy_reg = r10;
        auto addr_c_states_tm1_l_reg = rdi;
        auto addr_c_states_t_l_reg = rsi;
        // Here we cannot use rbp to have initial stack pointer so we
        // use rsp and offset it with the size of pushed registers in
        // preamble
        auto base_args = get_stack_params_address();
        mov(addr_states_t_l_copy_reg, ptr[base_args]);
        mov(addr_c_states_tm1_l_reg, ptr[base_args + 8]);
        mov(addr_c_states_t_l_reg, ptr[base_args + 16]);
        mov(addr_weights_peephole_reg, ptr[base_args + 24]);
        mov(n_step_reg, ptr[base_args + 40]);
#else
        auto addr_states_t_l_copy_reg = abi_param5;
        auto addr_c_states_tm1_l_reg = abi_param6;
        auto addr_c_states_t_l_reg = r10;
        auto base_args = get_stack_params_address();
        mov(addr_c_states_t_l_reg, ptr[base_args]);
        mov(addr_weights_peephole_reg, ptr[base_args + 8]);
        mov(n_step_reg, ptr[base_args + 24]);
#endif

        // helper lambda to address the gates and biases
        auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dhc * scratch_dt_size];
        };

        auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dhc * gate_dt_size];
        };
        auto weights_peephole_addr = [&](int i) {
            return ptr[addr_weights_peephole_reg
                    + i * rnn_.dhc * weights_peephole_dt_size];
        };
        auto B_addr = [&](int i) {
            return ptr[addr_bias_reg + i * rnn_.dhc * bias_dt_size];
        };

        // initialize registers with addresses and constants
        init_regs(weights_scales, vlen);

        sigmoid_injector_->load_table_addr();
        tanh_injector_->load_table_addr();
        if (rnn_.is_brgemm && !rnn_.unfused_post_gemm)
            mov(loop_cnt, n_step_reg);
        else
            mov(loop_cnt, rnn_.dhc * scratch_dt_size);
        cmp(loop_cnt, vlen);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L_aligned(vector_loop_start_label, 64);
        {
            const Vmm G0(1), G1(2), G2(3), G3(4);

            // load G0 G1 G2 G3
            uni_vmovups(G0, sg_addr(0));
            uni_vmovups(G1, sg_addr(1));
            uni_vmovups(G2, sg_addr(2));
            uni_vmovups(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed
            deq_w(src_data_t, G0, get_next_tmp_vmm(), get_next_tmp_vmm(),
                    0 * rnn_.dhc, mask, true);
            deq_w(src_data_t, G1, get_next_tmp_vmm(), get_next_tmp_vmm(),
                    1 * rnn_.dhc, mask, true);
            deq_w(src_data_t, G2, get_next_tmp_vmm(), get_next_tmp_vmm(),
                    2 * rnn_.dhc, mask, true);
            deq_w(src_data_t, G3, get_next_tmp_vmm(), get_next_tmp_vmm(),
                    3 * rnn_.dhc, mask, true);

            // add biases
            vaddps_rhs_op_mem(G0, G0, B_addr(0));
            vaddps_rhs_op_mem(G1, G1, B_addr(1));
            vaddps_rhs_op_mem(G2, G2, B_addr(2));
            vaddps_rhs_op_mem(G3, G3, B_addr(3));

            const auto tmp_c_states = get_next_tmp_vmm();
            uni_vmovups(tmp_c_states, ptr[addr_c_states_tm1_l_reg]);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                vfmadd231ps_rhs_op_mem(
                        G0, tmp_c_states, weights_peephole_addr(0));
                vfmadd231ps_rhs_op_mem(
                        G1, tmp_c_states, weights_peephole_addr(1));
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(0), G0, vlen);
                to_src<src_data_t>(wg_addr(1), G1, vlen);
                to_src<src_data_t>(wg_addr(2), G2, vlen);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmulps(tmp_c_states, tmp_c_states, G1);
            uni_vfmadd231ps(tmp_c_states, G0, G2);
            uni_vmovups(ptr[addr_c_states_t_l_reg], tmp_c_states);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                vfmadd231ps_rhs_op_mem(
                        G3, tmp_c_states, weights_peephole_addr(2));
            }

            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (is_training) { to_src<src_data_t>(wg_addr(3), G3, vlen); }

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp_c_states.getIdx());
            uni_vmulps(tmp_c_states, tmp_c_states, G3);

            // downconvert and write back the state
            to_src<src_data_t>(ptr[addr_states_t_l_reg], tmp_c_states, vlen);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, 0);
            je(vector_loop_inc_regs);
            to_src<src_data_t>(
                    ptr[addr_states_t_l_copy_reg], tmp_c_states, vlen, true);
            add(addr_states_t_l_copy_reg, vlen_dst);

            // increment address pointers
            L_aligned(vector_loop_inc_regs);
            add(addr_scratch_gates_reg, vlen);
            if (rnn_.is_lstm_peephole) add(addr_weights_peephole_reg, vlen);
            add(addr_bias_reg, vlen);
            add(addr_states_t_l_reg, vlen_dst);
            add(addr_c_states_tm1_l_reg, vlen);
            add(addr_c_states_t_l_reg, vlen);
            if (is_training) add(addr_ws_gates_reg, vlen_dst);
            inc_regs(mask, vlen);

            // increment loop counter
            sub(loop_cnt, vlen);
            cmp(loop_cnt, vlen);
            jge(vector_loop_start_label);
        }
        L_aligned(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);
        // Same code as above, we just use vmovss for accessing inputs
        L_aligned(rem_loop_start_label, 64);
        {
            const Xbyak::Xmm G0(1), G1(2), G2(3), G3(4);

            // load G0 G1 G2 G3
            uni_vmovss(G0, sg_addr(0));
            uni_vmovss(G1, sg_addr(1));
            uni_vmovss(G2, sg_addr(2));
            uni_vmovss(G3, sg_addr(3));

            // dequantize the gates from s32 to f32 if needed
            deq_w(src_data_t, G0, get_next_tmp_xmm(), get_next_tmp_xmm(),
                    0 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G1, get_next_tmp_xmm(), get_next_tmp_xmm(),
                    1 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G2, get_next_tmp_xmm(), get_next_tmp_xmm(),
                    2 * rnn_.dhc, mask, false);
            deq_w(src_data_t, G3, get_next_tmp_xmm(), get_next_tmp_xmm(),
                    3 * rnn_.dhc, mask, false);

            // add biases
            vaddss_rhs_op_mem(G0, G0, B_addr(0));
            vaddss_rhs_op_mem(G1, G1, B_addr(1));
            vaddss_rhs_op_mem(G2, G2, B_addr(2));
            vaddss_rhs_op_mem(G3, G3, B_addr(3));

            const auto tmp_c_states = get_next_tmp_xmm();
            uni_vmovss(tmp_c_states, ptr[addr_c_states_tm1_l_reg]);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                vfmadd231ss_rhs_op_mem(
                        G0, tmp_c_states, weights_peephole_addr(0));
                vfmadd231ss_rhs_op_mem(
                        G1, tmp_c_states, weights_peephole_addr(1));
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G0.getIdx());
            sigmoid_injector_->compute_vector(G1.getIdx());
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(G2.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(0), G0, scratch_dt_size);
                to_src<src_data_t>(wg_addr(1), G1, scratch_dt_size);
                to_src<src_data_t>(wg_addr(2), G2, scratch_dt_size);
            }

            // compute c_states_t_l = G1 * c_tm1_l + G0 * G2
            uni_vmulss(tmp_c_states, tmp_c_states, G1);
            uni_vfmadd231ss(tmp_c_states, G0, G2);
            uni_vmovss(ptr[addr_c_states_t_l_reg], tmp_c_states);

            // add peephole
            if (rnn_.is_lstm_peephole) {
                vfmadd231ss_rhs_op_mem(
                        G3, tmp_c_states, weights_peephole_addr(2));
            }

            // inject eltwise code
            sigmoid_injector_->load_table_addr();
            sigmoid_injector_->compute_vector(G3.getIdx());

            // if training we write back the gates
            if (is_training) {
                to_src<src_data_t>(wg_addr(3), G3, scratch_dt_size);
            }

            // states_t_l = G3 * tanh(c_states_t_l)
            tanh_injector_->load_table_addr();
            tanh_injector_->compute_vector(tmp_c_states.getIdx());
            uni_vmulss(tmp_c_states, tmp_c_states, G3);

            // downconcvert/quantize and write back the state
            to_src<src_data_t>(
                    ptr[addr_states_t_l_reg], tmp_c_states, scratch_dt_size);
            // if states_t_l_copy is a non null ptr, we write the output to it too
            cmp(addr_states_t_l_copy_reg, 0);
            je(rem_loop_inc_regs);
            to_src<src_data_t>(ptr[addr_states_t_l_copy_reg], tmp_c_states,
                    scratch_dt_size, true);
            add(addr_states_t_l_copy_reg, hstate_dt_size);

            // increment address pointers
            L_aligned(rem_loop_inc_regs);
            add(addr_scratch_gates_reg, scratch_dt_size);
            if (rnn_.is_lstm_peephole)
                add(addr_weights_peephole_reg, weights_peephole_dt_size);
            add(addr_bias_reg, bias_dt_size);
            add(addr_states_t_l_reg, hstate_dt_size);
            add(addr_c_states_tm1_l_reg, cstate_dt_size);
            add(addr_c_states_t_l_reg, cstate_dt_size);
            if (is_training) add(addr_ws_gates_reg, gate_dt_size);
            inc_regs(mask, qscale_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            cmp(loop_cnt, 0);
            jg(rem_loop_start_label);
        }
        L_aligned(rem_loop_end_label);

        postamble();

        sigmoid_injector_->prepare_table(true);
        tanh_injector_->prepare_table(true);

        init_table(vlen);
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
