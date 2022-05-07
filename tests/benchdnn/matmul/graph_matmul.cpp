/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "matmul/graph_matmul.hpp"

#include <algorithm>

namespace benchdnnext {
namespace matmul {

void check_known_skipped_case_graph(
        const ::matmul::prb_t *prb, res_t *res) noexcept {
    // TODO: to align with original benchdnn, we should consider moving
    // skip_unimplemented_prb call after compilation step
    skip_invalid_and_unimplemented_prb(prb, res);
    if (res->state == SKIPPED) return;

    check_graph_eltwise_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;
}

fill_status_t matmul_graph_prb_t::handle_main_op_(const ::matmul::prb_t *prb) {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);

    // this is needed to align with po_handlers convention
    // some patterns like `matmul + bias + swish` may want to
    // reuse bias output via `tensor_id["bias"].back() + "_DST"`
    if (has_post_bia_) tensor_id["bias"].push_back(TENSOR_ID);

    const auto orig_src_dt = convert_dt(prb->cfg[SRC].dt);
    const auto orig_wei_dt = convert_dt(prb->cfg[WEI].dt);
    const auto orig_dst_dt = convert_dt(prb->cfg[DST].dt);

    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string WEI {TENSOR_ID + "_WEI"};
    const std::string BIA {TENSOR_ID + "_BIA"};
    const std::string DST {TENSOR_ID + "_DST"};

    const auto is_lprec
            = is_low_precision({orig_src_dt, orig_wei_dt, orig_dst_dt});
    const auto with_tc = with_typecast({orig_src_dt, orig_wei_dt, orig_dst_dt});
    const auto change_dt = is_lprec || with_tc;
    const auto default_dt = (with_tc) ? dt::bf16 : dt::f32;
    const auto src_dt = (change_dt) ? default_dt : orig_src_dt;
    const auto wei_dt = (change_dt) ? default_dt : orig_wei_dt;
    const auto dst_dt = (change_dt) ? default_dt : orig_dst_dt;

    const auto src_dims
            = get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask());
    const auto wei_dims = get_runtime_dims(
            prb->weights_dims(), prb->weights_runtime_dim_mask());
    const auto dst_dims
            = get_runtime_dims(prb->dst_dims, prb->dst_runtime_dim_mask());

    tensor_descs_.emplace(SRC, src_dt, src_dims, prb->stag);
    tensor_descs_.emplace(WEI, wei_dt, wei_dims, prb->wtag,
            tensor_descs_t::property_type::constant);
    tensor_descs_.emplace(DST, dst_dt, dst_dims, prb->dtag);
    if (has_post_bia_) {
        std::vector<int64_t> bia_dims(dst_dims.size());
        for (int i = 0; i < prb->ndims; ++i)
            bia_dims[i] = (prb->bia_mask & (1 << i)) ? dst_dims[i] : 1;
        bia_dims = get_runtime_dims(bia_dims, prb->dst_runtime_dim_mask());
        tensor_descs_.emplace(BIA, convert_dt(prb->bia_dt), bia_dims,
                lt::strided, tensor_descs_t::property_type::constant);
    }

    std::vector<dnnl::graph::logical_tensor> lt_inputs {
            tensor_descs_[SRC], tensor_descs_[WEI]};
    std::vector<dnnl::graph::logical_tensor> lt_outputs {tensor_descs_[DST]};
    if (has_post_bia_) lt_inputs.push_back(tensor_descs_[BIA]);

    op matmul(new_op_id, op::kind::MatMul, lt_inputs, lt_outputs, "matmul");

    matmul.set_attr("transpose_a", false).set_attr("transpose_b", false);

    ops_.emplace_back(matmul);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t matmul_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.matmul.eltw_handler(*this, po_entry);
}

fill_status_t matmul_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.matmul.bin_handler(*this, po_entry);
}

fill_status_t matmul_graph_prb_t::handle_sum_() {
    return po_handler.matmul.sum_handler(*this);
}

fill_status_t matmul_graph_prb_t::handle_typecast_(const ::matmul::prb_t *prb) {
    using op = dnnl::graph::op;

    const std::string SRC = tensor_id["main"].back() + "_SRC";
    const std::string WEI = tensor_id["main"].back() + "_WEI";

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["typecast"].push_back(TENSOR_ID);
    const std::string TCSRC {TENSOR_ID + "_SRC"};
    const std::string TCWEI {TENSOR_ID + "_WEI"};

    tensor_descs_.emplace(TCSRC, dt::f32,
            get_runtime_dims(prb->src_dims(), prb->src_runtime_dim_mask()),
            prb->stag);
    tensor_descs_.emplace(TCWEI, dt::f32,
            get_runtime_dims(
                    prb->weights_dims(), prb->weights_runtime_dim_mask()),
            prb->wtag);

    op typecast_src(ops_.size(), op::kind::TypeCast, {tensor_descs_[TCSRC]},
            {tensor_descs_[SRC]}, "typecast_src");
    ops_.emplace_back(typecast_src);
    op typecast_wei(ops_.size(), op::kind::TypeCast, {tensor_descs_[TCWEI]},
            {tensor_descs_[WEI]}, "typecast_wei");
    ops_.emplace_back(typecast_wei);

    return fill_status_t::DONE;
}

fill_status_t matmul_graph_prb_t::handle_low_precision_(
        const ::matmul::prb_t *prb) {
    const auto src_dt = convert_dt(prb->cfg[SRC].dt);
    const auto wei_dt = convert_dt(prb->cfg[WEI].dt);
    const auto dst_dt = convert_dt(prb->cfg[DST].dt);
    const bool with_tc = with_typecast({src_dt, wei_dt, dst_dt});
    const bool def_oscales = prb->attr.oscale.is_def();

    // currently, only policy_t::COMMON is supported for asymmetric quant
    // for src and dst, other policy is not suppoted by oneDNN Graph.
    // zps for src
    const int64_t common_zp_count = 1;
    const int64_t dflt_zp_val = 0;
    src_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb->attr.zero_points.is_def(DNNL_ARG_SRC)) {
        const auto &src_zp_e = prb->attr.zero_points.get(DNNL_ARG_SRC);
        if (src_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        src_zero_points[0] = prb->src_zp[0];
    }
    // zps for wei
    wei_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb->attr.zero_points.is_def(DNNL_ARG_WEIGHTS)) {
        const auto &wei_zp_e = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS);
        if (wei_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        wei_zero_points[0] = wei_zp_e.value;
    }
    // zps for dst
    dst_zero_points.resize(common_zp_count, dflt_zp_val);
    // if zp is not default, copy values and pass it to oneDNN Graph
    if (!prb->attr.zero_points.is_def(DNNL_ARG_DST)) {
        const auto &dst_zp_e = prb->attr.zero_points.get(DNNL_ARG_DST);
        if (dst_zp_e.policy != policy_t::COMMON)
            return fill_status::UNSUPPORTED_CONFIG;
        dst_zero_points[0] = prb->dst_zp[0];
    }

    const float common_scale = [&prb, this]() {
        if (has_post_eltwise()) {
            const float post_eltwise_scale
                    = get_post_eltwise_scale(prb->attr.post_ops.entry);
            // benchdnn ext. need to convert post relu scale to quant scale to
            // get same result as benchdnn primitive did
            return 1.f * (1 / post_eltwise_scale);
        } else {
            return 1.f;
        }
    }();

    low_precision_attr lp_attr = low_precision_attr::lp_attr(src_dt, wei_dt,
            dst_dt, prb->stag, prb->wtag, prb->dtag, prb->attr.oscale.policy,
            &oscales_, common_scale, &src_zero_points, &wei_zero_points,
            &dst_zero_points, prb->scales, prb->n, def_oscales, with_tc);

    fill_status_t ctor_status;
    ctor_status
            = po_handler.matmul.low_precision_handler.handle_low_precision_src(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.matmul.low_precision_handler.handle_low_precision_wei(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    // `with_qdst == false` means that we are dealing
    // with Quantized lacking pattern, like x8s8f32 or x8s8bf16
    const bool with_qdst = dt::u8 == dst_dt || dt::s8 == dst_dt;
    if (with_qdst) {
        ctor_status = po_handler.matmul.low_precision_handler
                              .handle_low_precision_dst(*this, lp_attr);
    }
    if (ctor_status != fill_status::DONE) return ctor_status;

    if (has_post_sum()) {
        ctor_status = po_handler.matmul.low_precision_handler
                              .handle_low_precision_post_sum(
                                      *this, lp_attr, prb->attr.post_ops.entry);
    }

    return ctor_status;
}

dims_t get_runtime_dims(
        const dims_t &dims, const ::matmul::dims_mask_t &mask) noexcept {
    if (mask.none() || dims.empty()) return dims;
    dims_t runtime_dims;
    runtime_dims.resize(dims.size());
    const int64_t axis_unknown_flag = -1;
    for (size_t i = 0; i < dims.size(); ++i) {
        runtime_dims[i] = mask[i] ? axis_unknown_flag : dims[i];
    }
    return runtime_dims;
}

int doit(const ::matmul::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    matmul_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filter partitions
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::matmul::init_pd, prb, res, par, ins, outs);

    const auto apply_bias = convert_dt(prb->bia_dt) != dt::undef;

    size_t idx_ins = 0;
    auto src_fp = make_dnn_mem(ins[idx_ins], dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(ins[idx_ins], prb->stag);
    auto wei_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx);
    auto wei_dt = make_dnn_mem(ins[idx_ins], prb->wtag);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
    auto dst_dt = make_dnn_mem(outs[0], prb->dtag);
    dnn_mem_t bia_fp, bia_dt;
    if (apply_bias) {
        bia_fp = make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx);
        bia_dt = make_dnn_mem(ins[idx_ins], tag::abx);
    }

    SAFE(fill_data(SRC, prb, src_dt, src_fp, res), WARN);
    SAFE(fill_data(WEI, prb, wei_dt, wei_fp, res), WARN);
    SAFE(fill_data(DST, prb, dst_dt, dst_fp, res), WARN);
    if (apply_bias) SAFE(fill_data(BIA, prb, bia_dt, bia_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    const std::vector<attr_t::post_ops_t::entry_t> &po_entry
            = prb->attr.post_ops.entry;
    const std::vector<size_t> post_bin_indices = get_post_bin_indices(po_entry);
    if (graph_prb.has_post_bin()) {
        for (size_t i = 0; i < post_bin_indices.size(); i++) {
            binary_po_fp.emplace_back(
                    make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx));
            binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], tag::abx));
            binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                     static_cast<int>(post_bin_indices[i])),
                    binary_po_dt[i], binary_po_fp[i]);
        }
    }
    const dnnl::graph::engine &eng = get_test_engine();

    idx_ins = 0;
    dnnl::graph::tensor src_tensor(
            ins[idx_ins], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(
            ins[++idx_ins], eng, static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));
    dnnl::graph::tensor bia_tensor;
    dnnl::graph::tensor bin_tensor;
    dnnl::graph::tensor sum_src1_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor, wei_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    if (apply_bias) {
        bia_tensor = dnnl::graph::tensor(
                ins[++idx_ins], eng, static_cast<void *>(bia_dt));
        tensors_in.emplace_back(bia_tensor);
    }

    size_t bin_dt_idx = 0;
    for (size_t i = 0; i < po_entry.size(); i++) {
        // we can't have fuse with both sum and binary-add at the same time
        if (po_entry[i].is_sum_kind()) { // Always use in-place operation.
            sum_src1_tensor = dnnl::graph::tensor(
                    ins[++idx_ins], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry[i].is_binary_kind()) {
            bin_tensor = dnnl::graph::tensor(ins[++idx_ins], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_tensor);
            ++bin_dt_idx;
        }
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    dnn_mem_t bia_fp_scaled;
    args_t args, ref_args;

    if (is_bench_mode(CORR)) {
        args.set(DNNL_ARG_DST, dst_dt);
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);

        std::vector<int> binary_po_args;
        for (size_t idx_bin : post_bin_indices) {
            binary_po_args.emplace_back(
                    (DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(idx_bin))
                            | DNNL_ARG_SRC_1));
        }
        ref_args.set(binary_po_args, binary_po_fp);

        if (apply_bias
                && is_low_precision({convert_dt(prb->cfg[SRC].dt),
                        convert_dt(prb->cfg[WEI].dt),
                        convert_dt(prb->cfg[DST].dt)})) {
            bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::abx);
            scale_bia(bia_fp_scaled, bia_fp, graph_prb.get_oscales());
            ref_args.set(DNNL_ARG_BIAS, bia_fp_scaled);
        } else {
            ref_args.set(DNNL_ARG_BIAS, bia_fp);
        }

        check_correctness(prb, {DST}, args, ref_args, ::matmul::setup_cmp, res);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace matmul
} // namespace benchdnnext
