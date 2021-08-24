/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GRAPH_CONV_HPP
#define GRAPH_CONV_HPP

#include <vector>

#include "conv/conv.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace conv {

struct conv_graph_prb_t : public graph_prb_t {
    conv_graph_prb_t(const ::conv::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;
        if (prb->dir == FWD_B) {
            has_post_bia_ = true;
            ctor_status = handle_bia_();
            if (stop_work(ctor_status)) return;
        }

        const std::vector<attr_t::post_ops_t::entry_t> &po_entry
                = prb->attr.post_ops.entry;

        for (const attr_t::post_ops_t::entry_t &po : po_entry) {
            if (po.is_eltwise_kind()) {
                ctor_status = handle_elt_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_sum_kind()) {
                has_post_sum_ = true;
                ctor_status = handle_sum_();
                if (stop_work(ctor_status)) return;
            } else if (po.is_binary_kind()) {
                has_post_bin_ = true;
                ctor_status = handle_bin_(po);
                if (stop_work(ctor_status)) return;
            }
        }

        auto dtypes = {spec_.src_dt, spec_.dst_dt};
        if (benchdnnext::is_low_precision(dtypes)) {
            ctor_status = handle_low_precision_(prb);
            if (stop_work(ctor_status)) return;
        }

        ctor_status = fill_status::DONE;
    };
    fill_status_t ctor_status;

private:
    std::vector<float> oscales;
    struct spec_t {
        spec_t(const ::conv::prb_t *prb) noexcept;

        dims_t src_dims;
        dims_t wei_dims;
        dims_t bia_dims;
        dims_t dst_dims;

        dims_t strides;
        dims_t pads_begin;
        dims_t pads_end;
        dims_t dilations;

        std::string auto_pad {"None"};

        bool has_groups;
        int64_t groups;

        std::string data_format {"NCX"};
        std::string filter_format {"OIX"};
        std::string raw_src_tag;
        std::string raw_wei_tag;
        std::string raw_dst_tag;

        dt src_dt;
        dt wei_dt;
        dt bia_dt;
        dt dst_dt;
    };

    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_bia_();
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_sum_();
    fill_status_t handle_low_precision_(const ::conv::prb_t *prb);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::Convolution;
    }

public:
    const spec_t &spec() const noexcept { return spec_; }

    std::vector<float> get_oscales() noexcept { return oscales; }
};

int doit(const ::conv::prb_t *prb, res_t *res);

} // namespace conv
} // namespace benchdnnext

#endif
