/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_INNER_PRODUCT_HPP
#define CPU_X64_BRGEMM_INNER_PRODUCT_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_brgemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
static const int max_num_brg_kernels_ip = 2 * 2 * 2 * 2;

inline int get_brg_kernel_index(const jit_brgemm_primitive_conf_t &jbgp,
        bool do_initialization, bool is_M_tail, bool is_N_tail,
        bool is_K_tail) {
    auto vM = (is_M_tail) ? jbgp.M_tail : jbgp.M;
    auto vN = (is_N_tail) ? jbgp.N_tail : jbgp.N;
    auto vK = (is_K_tail) ? jbgp.K_tail : jbgp.K;
    if (vM == 0 || vN == 0 || vK == 0 || jbgp.LDA < vK || jbgp.LDB < vN
            || jbgp.LDC < vN)
        return -1;

    int idx = 8 * (int)do_initialization + 4 * (int)is_M_tail
            + 2 * (int)is_N_tail + (int)is_K_tail;

    assert(idx < max_num_brg_kernels_ip);
    return idx;
}

} // namespace

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type>
struct brgemm_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgemm:", avx512_core, ""),
                brgemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            auto check_attr = [=]() {
                if (utils::one_of(src_type, data_type::u8, data_type::s8)) {
                    return attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops);
                } else {
                    return attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops);
                }
            };

            bool ok = true && mayiuse(avx512_core) && is_fwd()
                    && expect_data_types(src_type, wei_type, data_type::undef,
                            dst_type, data_type::undef)
                    && IMPLICATION(with_bias(),
                            ((utils::one_of(
                                      src_type, data_type::u8, data_type::s8)
                                     && utils::one_of(bias_md_.data_type,
                                             data_type::f32, data_type::s32,
                                             data_type::s8, data_type::u8))
                                    || (utils::one_of(src_type, data_type::bf16)
                                            && utils::one_of(bias_md_.data_type,
                                                    data_type::f32,
                                                    data_type::bf16))
                                    || utils::everyone_is(data_type::f32,
                                            src_type, bias_md_.data_type)))
                    && check_attr() && !has_zero_dim_memory()
                    && set_default_formats();
            if (!ok) return status::unimplemented;

            CHECK(brgemm_inner_product_utils::init_ip_conf(jbgp_, *desc(),
                    *src_md(), *weights_md(), *dst_md(), *attr(),
                    dnnl_get_max_threads()));

            const float alpha = 1.0;
            const float beta = 1.0;
            const float beta_init = 0.0;
            for_(int i_init = 0; i_init < 2; i_init++)
            for_(int i_M = 0; i_M < 2; i_M++)
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_K = 0; i_K < 2; i_K++) {
                auto vbeta = (i_init) ? beta_init : beta;
                auto vM = (i_M) ? jbgp_.M_tail : jbgp_.M;
                auto vN = (i_N) ? jbgp_.N_tail : jbgp_.N;
                auto vK = (i_K) ? jbgp_.K_tail : jbgp_.K;

                int idx = get_brg_kernel_idx(i_init, i_M, i_N, i_K);
                if (idx < 0) continue;
                brgemm_t &brg = brg_descs_[idx];
                CHECK(brgemm_desc_init(&brg, isa_any, jbgp_.brg_type, src_type,
                        wei_type, false, false, brgemm_row_major, alpha, vbeta,
                        jbgp_.LDA, jbgp_.LDB, jbgp_.LDC, vM, vN, vK));

                auto dt_d = dst_type;
                auto LDD = jbgp_.oc_without_padding;
                CHECK(brgemm_desc_add_postops(
                        &brg, attr(), dt_d, LDD, jbgp_.bia_dt));
            }

            auto scratchpad = scratchpad_registry().registrar();
            brgemm_inner_product_utils::init_scratchpad(scratchpad, jbgp_);

            return status::success;
        }

        int get_brg_kernel_idx(bool do_initialization, bool is_M_tail,
                bool is_N_tail, bool is_K_tail) const {
            return get_brg_kernel_index(
                    jbgp_, do_initialization, is_M_tail, is_N_tail, is_K_tail);
        }

        brgemm_t brg_descs_[max_num_brg_kernels_ip];
        jit_brgemm_primitive_conf_t jbgp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = nc;
            auto wei_tag
                    = brgemm_inner_product_utils::get_brgemm_ip_weights_tag(
                            OC(), invariant_wei_md()->data_type);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    brgemm_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        for_(int i_M = 0; i_M < 2; i_M++)
        for_(int i_N = 0; i_N < 2; i_N++)
        for_(int i_K = 0; i_K < 2; i_K++)
        for (int i_init = 0; i_init < 2; i_init++) {
            int idx = pd()->get_brg_kernel_idx(i_init, i_M, i_N, i_K);
            if (idx < 0) continue;

            brgemm_kernel_t *ker = nullptr;
            CHECK(brgemm_kernel_create(&ker, pd()->brg_descs_[idx]));
            CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    std::unique_ptr<brgemm_kernel_t> brg_kernels_[max_num_brg_kernels_ip];
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
