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

#include "conv/graph_conv_dw_fusion.hpp"
#include "conv/conv_dw_fusion.hpp"
#include "conv/graph_conv.hpp"

namespace benchdnnext {
namespace conv_dw_fusion {

namespace graph = dnnl::graph;

void check_known_skipped_case_graph(
        const ::conv_dw_fusion::prb_t *prb, res_t *res) noexcept {
    ::conv_dw_fusion::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    // We do not support backward pass and bias at the moment
    if (prb->dir != FWD_I && prb->dir != FWD_D) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

// Current filling doesn't work for fused_wei due to relying on prb values,
// which are different for fused conv. Therefore, the current solution
// is to create and fill the first convolution and the depthwise convolution separately.
// Then create the original conv+dw problem and fill it with data from previously
// created separately convolutions.
// TODO: When the filling will be fixed - change the infrastructure.
int doit(const ::conv_dw_fusion::prb_t *prb, res_t *res) {
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    // Fill first convolution
    std::unique_ptr<::conv_dw_fusion::prb_t> p0
            = ::conv_dw_fusion::get_first_conv_prb(prb);
    if (!p0) SAFE(FAIL, CRIT);
    conv::conv_graph_prb_t graph_prb0(p0.get());

    if (graph_prb0.ctor_status != fill_status::DONE
            && graph_prb0.ctor_status
                    != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h0 = graph_prb0.to_graph();
    const auto spec_conv = graph_prb0.spec();

    // Filter partitions
    const auto partitions0
            = graph_h0.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions0.empty() || partitions0.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par0 = partitions0[0];
    if (!par0.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins0 = par0.get_in_ports();
    const auto outs0 = par0.get_out_ports();

    auto init_pd = [&](dnnl_engine_t engine, const ::conv_dw_fusion::prb_t *prb,
                           dnnl_primitive_desc_t &cpd, res_t *res, dir_t dir,
                           const_dnnl_primitive_desc_t hint) {
        SAFE(::conv_dw_fusion::init_pd(engine, prb, cpd, res, dir, hint), WARN);
        return OK;
    };
    auto cp0 = compile_partition(init_pd, p0.get(), res, par0, ins0, outs0);

    auto src_fp0 = make_dnn_mem(ins0[0], spec_conv.src_dims, dt::f32, tag::abx);
    auto wei_fp0 = make_dnn_mem(ins0[1], spec_conv.wei_dims, dt::f32, tag::abx);

    dnn_mem_t bia_fp0;
    if (prb->dir == FWD_B) bia_fp0 = make_dnn_mem(ins0[2], dt::f32, tag::x);
    auto dst_fp0
            = make_dnn_mem(outs0[0], spec_conv.dst_dims, dt::f32, tag::abx);

    auto src_dt0
            = make_dnn_mem(ins0[0], spec_conv.src_dims, spec_conv.raw_src_tag);
    auto wei_dt0
            = make_dnn_mem(ins0[1], spec_conv.wei_dims, spec_conv.raw_wei_tag);
    dnn_mem_t bia_dt0;
    if (prb->dir == FWD_B) bia_dt0 = make_dnn_mem(ins0[2], tag::x);
    auto dst_dt0
            = make_dnn_mem(outs0[0], spec_conv.dst_dims, spec_conv.raw_dst_tag);

    SAFE(::conv::fill_src(p0.get(), src_dt0, src_fp0, res), WARN);
    SAFE(::conv::fill_wei(p0.get(), wei_dt0, wei_fp0, res), WARN);
    SAFE(::conv::fill_bia(p0.get(), bia_dt0, bia_fp0, res), WARN);
    SAFE(::conv::fill_dst(p0.get(), dst_dt0, dst_fp0, res), WARN);

    // Fill depthwise convolution
    std::unique_ptr<::conv_dw_fusion::prb_t> p1
            = ::conv_dw_fusion::get_fused_conv_prb(prb);
    if (!p1) SAFE(FAIL, CRIT);
    conv::conv_graph_prb_t graph_prb1(p1.get());

    if (graph_prb1.ctor_status != fill_status::DONE
            && graph_prb1.ctor_status
                    != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h1 = graph_prb1.to_graph();
    const auto spec_dw = graph_prb1.spec();

    // Filter partitions
    const auto partitions1
            = graph_h1.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions1.empty() || partitions1.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par1 = partitions1[0];
    if (!par1.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins1 = par1.get_in_ports();
    const auto outs1 = par1.get_out_ports();

    auto cp1 = compile_partition(init_pd, p1.get(), res, par1, ins1, outs1);

    auto wei_fp1 = make_dnn_mem(ins1[1], spec_dw.wei_dims, dt::f32, tag::abx);

    dnn_mem_t bia_fp1;
    if (prb->dir == FWD_B) bia_fp1 = make_dnn_mem(ins1[2], dt::f32, tag::x);
    auto dst_fp1 = make_dnn_mem(outs1[0], spec_dw.dst_dims, dt::f32, tag::abx);

    auto src_dt1 = make_dnn_mem(ins1[0], spec_dw.src_dims, spec_dw.raw_src_tag);
    auto wei_dt1 = make_dnn_mem(ins1[1], spec_dw.wei_dims, spec_dw.raw_wei_tag);
    dnn_mem_t bia_dt1;
    if (prb->dir == FWD_B) bia_dt1 = make_dnn_mem(ins1[2], tag::x);
    auto dst_dt1
            = make_dnn_mem(outs1[0], spec_dw.dst_dims, spec_dw.raw_dst_tag);

    SAFE(::conv::fill_wei(p1.get(), wei_dt1, wei_fp1, res), WARN);
    SAFE(::conv::fill_bia(p1.get(), bia_dt1, bia_fp1, res), WARN);
    SAFE(::conv::fill_dst(p1.get(), dst_dt1, dst_fp1, res), WARN);

    // Original problem with fusion attributes
    conv::conv_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec_fused = graph_prb.spec();

    // Filter partitions
    const auto partitions
            = graph_h.get_partitions(dnnl::graph::partition::policy::fusion);
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], spec_fused.src_dims, dt::f32, tag::abx);
    auto wei_fp = make_dnn_mem(ins[1], spec_fused.wei_dims, dt::f32, tag::abx);
    dnn_mem_t bia_fp;
    if (prb->dir == FWD_B) bia_fp = make_dnn_mem(ins[2], dt::f32, tag::x);
    auto dst_fp = make_dnn_mem(outs[0], spec_dw.dst_dims, dt::f32, tag::abx);
    auto fused_wei_fp
            = make_dnn_mem(ins.back(), spec_dw.wei_dims, dt::f32, tag::abx);
    dnn_mem_t fused_bia_fp;
    if (prb->dir == FWD_B)
        fused_bia_fp = make_dnn_mem(ins.back(), dt::f32, tag::x);

    auto src_dt
            = make_dnn_mem(ins[0], spec_fused.src_dims, spec_fused.raw_src_tag);
    auto wei_dt
            = make_dnn_mem(ins[1], spec_fused.wei_dims, spec_fused.raw_wei_tag);
    dnn_mem_t bia_dt;
    if (prb->dir == FWD_B) bia_dt = make_dnn_mem(ins[2], tag::x);
    auto dst_dt = make_dnn_mem(outs[0], spec_dw.dst_dims, spec_dw.raw_dst_tag);
    auto fused_wei_dt = make_dnn_mem(
            ins.back(), spec_dw.wei_dims, spec_fused.raw_wei_tag);
    dnn_mem_t fused_bia_dt;
    if (prb->dir == FWD_B) fused_bia_dt = make_dnn_mem(ins.back(), tag::x);

    // Work around for the filling issue
    SAFE(src_dt.reorder(src_fp0), WARN);
    SAFE(wei_dt.reorder(wei_fp0), WARN);
    if (prb->dir == FWD_B) SAFE(bia_dt.reorder(bia_fp0), WARN);
    SAFE(dst_dt.reorder(dst_fp1), WARN);
    SAFE(fused_wei_dt.reorder(wei_fp1), WARN);
    if (prb->dir == FWD_B) SAFE(fused_bia_dt.reorder(bia_fp1), WARN);

    dnnl::graph::engine &eng = get_test_engine();

    // Execute first convolution
    graph::tensor src_tensor0(ins0[0], eng, static_cast<void *>(src_dt0));
    graph::tensor wei_tensor0(ins0[1], eng, static_cast<void *>(wei_dt0));
    graph::tensor bia_tensor0;
    if (prb->dir == FWD_B)
        bia_tensor0 = graph::tensor(ins0[2], eng, static_cast<void *>(bia_dt0));
    graph::tensor dst_tensor0(outs0[0], eng, static_cast<void *>(dst_dt0));

    std::vector<graph::tensor> input_ts0 {src_tensor0, wei_tensor0};
    if (prb->dir == FWD_B) input_ts0.push_back(bia_tensor0);

    std::vector<graph::tensor> output_ts0 {dst_tensor0};

    SAFE(execute_and_wait(cp0, input_ts0, output_ts0), WARN);
    SAFE(src_dt1.reorder(dst_dt0), WARN);

    // Execute depthwise convolution
    graph::tensor src_tensor1(ins1[0], eng, static_cast<void *>(src_dt1));
    graph::tensor wei_tensor1(ins1[1], eng, static_cast<void *>(wei_dt1));
    graph::tensor bia_tensor1;
    if (prb->dir == FWD_B)
        bia_tensor1 = graph::tensor(ins1[2], eng, static_cast<void *>(bia_dt1));
    graph::tensor dst_tensor1(outs1[0], eng, static_cast<void *>(dst_dt1));

    std::vector<graph::tensor> input_ts1 {src_tensor1, wei_tensor1};
    if (prb->dir == FWD_B) input_ts1.push_back(bia_tensor1);

    std::vector<graph::tensor> output_ts1 {dst_tensor1};

    SAFE(execute_and_wait(cp1, input_ts1, output_ts1), WARN);

    // Original problem with fusion attributes
    graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    graph::tensor wei_tensor(ins[1], eng, static_cast<void *>(wei_dt));
    graph::tensor bia_tensor;
    if (prb->dir == FWD_B)
        bia_tensor = graph::tensor(ins[2], eng, static_cast<void *>(bia_dt));
    graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    graph::tensor fused_wei_tensor(
            ins.back(), eng, static_cast<void *>(fused_wei_dt));
    graph::tensor fused_bia_tensor;
    if (prb->dir == FWD_B)
        fused_bia_tensor = graph::tensor(
                ins.back(), eng, static_cast<void *>(fused_bia_dt));

    std::vector<graph::tensor> input_ts {src_tensor, wei_tensor};
    if (prb->dir == FWD_B) input_ts.push_back(bia_tensor);
    input_ts.push_back(fused_wei_tensor);
    if (prb->dir == FWD_B) input_ts.push_back(fused_bia_tensor);

    std::vector<graph::tensor> output_ts {dst_tensor};

    SAFE(execute_and_wait(cp, input_ts, output_ts), WARN);

    if (is_bench_mode(CORR)) {
        const auto fp = dnnl_f32;
        const auto &dnnl_test_engine = ::get_test_engine();
        dnn_mem_t dst_fused(dst_dt, fp, tag::abx, dnnl_test_engine);
        dnn_mem_t dst_unfused(dst_dt1, fp, tag::abx, dnnl_test_engine);

        SAFE(::conv::compare_data(p1.get(), DST, dst_fused, dst_unfused, res),
                WARN);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, input_ts, output_ts),
            WARN);

    return OK;
}

} // namespace conv_dw_fusion
} // namespace benchdnnext
