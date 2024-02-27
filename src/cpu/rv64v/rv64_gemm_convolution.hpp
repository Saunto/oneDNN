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

#include "cpu/rv64v/jit/convolution/driver.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <data_type_t T>
struct rv64_gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("rv64:gemm", rv64_gemm_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            bool ok = is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(T, T, data_type::undef, T, T)
                && platform::has_data_type_support(T)
                && IMPLICATION(with_bias(), bias_md_.data_type == T)
                && attr()->has_default_values(smask_t::oscale
                        | smask_t::zero_points_runtime
                        | smask_t::post_ops,
                        T)
                && attr()->post_ops_.len() == 0
                && init_conf(conf_,
                    desc_, dst_md_, src_md_, weights_md_, bias_md_)
                && pick_memory_formats_from_conf(conf_, dst_md_, src_md_,
                    weights_md_, bias_md_);
            return ok ? status::success : status::unimplemented;
        }

        /// The convolution kernel arguments and optimizations structure
        jit_convolution_configuration_t conf_;
    };

    rv64_gemm_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<T>::type data_t;

    status_t init(engine_t *engine) override {
        init_schedule(schedule, pd()->conf_);
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return do_execute(ctx);
    }

private:
    convolution_schedule_t schedule;
    status_t do_execute(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};


} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_RV64_CONVOLUTION_HPP

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
