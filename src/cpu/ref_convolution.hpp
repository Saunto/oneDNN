/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_REF_CONVOLUTION_HPP
#define CPU_REF_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type,
        impl::data_type_t acc_type = dst_type>
struct ref_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, data_type::undef,
                            dst_type, acc_type)
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(wei_type)
                    && platform::has_data_type_support(dst_type)
                    && IMPLICATION(with_bias(),
                            true
                                    && IMPLICATION(src_type == u8,
                                            utils::one_of(bias_md_.data_type,
                                                    f32, s32, s8, u8))
                                    && IMPLICATION(src_type == f32,
                                            bias_md_.data_type == f32))
                    && set_default_formats()
                    && attr()->has_default_values(smask_t::oscale
                                    | smask_t::zero_points_runtime
                                    | smask_t::post_ops,
                            dst_type)
                    && output_scales_mask_ok() && zero_points_ok()
                    && post_ops_ok();
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool output_scales_mask_ok() const {
            using namespace data_type;
            const auto &mask = attr()->output_scales_.mask_;
            return IMPLICATION(!utils::one_of(src_type, s8, u8),
                           attr()->output_scales_.has_default_values())
                    && (mask == 0 || mask == 1 << 1);
        }

        bool zero_points_ok() const {
            using namespace data_type;
            int mask_src = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, &mask_src, nullptr);
            attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &mask_dst, nullptr);

            return IMPLICATION(!utils::one_of(src_type, s8, u8),
                           attr()->zero_points_.has_default_values())
                    && attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && (mask_src == 0 || mask_src == 1 << 1)
                    && (mask_dst == 0 || mask_dst == 1 << 1);
        }

        bool post_ops_ok() const {
            // to be consistent with other primitives and documentation
            // the number and sequence of post op is limited
            using namespace dnnl::impl::primitive_kind;
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            switch (po.len()) {
                case 0: return true;
                case 1: return is_eltwise(0) || po.contain(sum, 0);
                case 2:
                    return (po.contain(sum, 0) && is_eltwise(1))
                            || (po.contain(sum, 1) && is_eltwise(0));
                default: return false;
            }
            return false;
        }
    };

    ref_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace primitive_kind;
        const auto &po = pd()->attr()->post_ops_;
        for (auto idx = 0; idx < po.len(); ++idx) {
            if (po.contain(eltwise, idx))
                eltwise_ker_.push_back(
                        utils::make_unique<ref_eltwise_scalar_fwd_t>(
                                po.entry_[idx].eltwise));
        }
        return status::success;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<std::unique_ptr<ref_eltwise_scalar_fwd_t>> eltwise_ker_;
};

template <impl::data_type_t diff_src_type, impl::data_type_t wei_type,
        impl::data_type_t diff_dst_type,
        impl::data_type_t acc_type = diff_src_type>
struct ref_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_type, wei_type,
                            data_type::undef, diff_dst_type, acc_type)
                    && platform::has_data_type_support(diff_src_type)
                    && platform::has_data_type_support(wei_type)
                    && platform::has_data_type_support(diff_dst_type)
                    && set_default_formats()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale)
                    && output_scales_mask_ok();

            return ok ? status::success : status::unimplemented;
        }

        bool support_bias() const override { return true; }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool output_scales_mask_ok() const {
            using namespace data_type;
            const auto &mask = attr()->output_scales_.mask_;
            return IMPLICATION(!utils::one_of(diff_dst_type, s8, u8),
                           attr()->output_scales_.has_default_values())
                    && (mask == 0 || mask == 1 << 1);
        }
    };

    ref_convolution_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <impl::data_type_t src_type, impl::data_type_t diff_wei_type,
        impl::data_type_t diff_dst_type,
        impl::data_type_t acc_type = diff_wei_type>
struct ref_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, diff_wei_type, diff_wei_type,
                            diff_dst_type, acc_type)
                    && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(diff_wei_type)
                    && platform::has_data_type_support(diff_dst_type)
                    && set_default_formats() && attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_bwd_weights_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_wei_type>::type diff_wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
