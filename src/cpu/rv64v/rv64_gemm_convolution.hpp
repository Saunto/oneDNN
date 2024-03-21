#ifndef CPU_RV64V_RV64_GEMM_CONVOLUTION_HPP
#define CPU_RV64V_RV64_GEMM_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/gemm_convolution_utils.hpp"


namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <data_type_t T>
struct rv64_gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}
        
        //using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;
        DECLARE_COMMON_PD_T("rv64:gemm", rv64_gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(f32, f32, f32, f32, f32)
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32)
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };
            auto is_binary
                    = [&](int idx) { return po.entry_[idx].is_binary(); };

            for (int idx = 0; idx < po.len(); idx++) {
                bool ok = utils::one_of(true, is_sum(idx), is_binary(idx),
                                  is_eltwise(idx))
                        && IMPLICATION(is_sum(idx), idx == 0);
                if (!ok) return false;
            }

            return true;
        }
    };

    rv64_gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), post_ops_(nullptr) {}

    status_t init(engine_t *engine) override {
        const data_t one = 1.0, zero = 0.0;
        const auto &jcp = pd()->jcp_;
        beta_ = jcp.with_sum ? one : zero;

        if (jcp.with_eltwise || jcp.with_binary)
            CHECK(safe_ptr_assign(post_ops_, new ref_post_ops_t(jcp.post_ops)));
        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_forward_nspc(ctx) : execute_forward_ncsp(ctx);
    }

private:
    status_t execute_forward_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_forward_nspc(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr_nspc(const exec_ctx_t &ctx, const int ithr,
            const int nthr, const data_t *src_base, const data_t *wei_base,
            const data_t *bia_base, data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    data_t beta_;

    std::unique_ptr<ref_post_ops_t> post_ops_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif