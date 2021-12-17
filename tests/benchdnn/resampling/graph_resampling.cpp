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
#include <ctime>
#include <random>
#include "binary/binary.hpp"
#include "resampling/graph_resampling.hpp"

namespace benchdnnext {
namespace resampling {

resampling_graph_prb_t::spec_t::spec_t(
        const ::resampling::prb_t *prb) noexcept {
    switch (prb->ndims) {
        case 5:
            src_dims = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
            dst_dims = {prb->mb, prb->ic, prb->od, prb->oh, prb->ow};
            sizes = {prb->od, prb->oh, prb->ow};
            scales = {(float)prb->od / prb->id, (float)prb->oh / prb->ih,
                    (float)prb->ow / prb->iw};
            break;
        case 4:
            src_dims = {prb->mb, prb->ic, prb->ih, prb->iw};
            dst_dims = {prb->mb, prb->ic, prb->oh, prb->ow};
            sizes = {prb->oh, prb->ow};
            scales = {(float)prb->oh / prb->ih, (float)prb->ow / prb->iw};
            break;
        case 3:
            src_dims = {prb->mb, prb->ic, prb->iw};
            dst_dims = {prb->mb, prb->ic, prb->ow};
            sizes = {prb->ow};
            scales = {(float)prb->ow / prb->iw};
            break;
        default: assert("unknown dims size");
    }
    src_dt = convert_dt(prb->sdt);
    dst_dt = convert_dt(prb->ddt);
    tag = prb->tag;
    mode = alg2str(prb->alg);

    srand(std::time(NULL));
    rand_testmode = (rand() % 2);
}

fill_status_t resampling_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string SIZES {TENSOR_ID + "_SIZES"};
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(SRC, spec_.src_dt, spec_.src_dims, spec_.tag);
    if (spec_.rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
        tensor_descs_.emplace(SIZES, dt::s32, {1}, spec_.tag);
    }
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, spec_.tag);

    std::vector<dnnl::graph::logical_tensor> lt_inputs {tensor_descs_[SRC]};
    std::vector<dnnl::graph::logical_tensor> lt_outputs {tensor_descs_[DST]};
    if (spec_.rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
        lt_inputs.emplace_back(tensor_descs_[SIZES]);
    }

    op resampling(new_op_id, op::kind::Interpolate, lt_inputs, lt_outputs,
            "interpolate");
    resampling.set_attr("mode", spec_.mode);
    resampling.set_attr("data_format", spec_.data_format);
    if (spec_.rand_testmode == test_mode_t::SIZES_ATTR) {
        resampling.set_attr<std::vector<int64_t>>("sizes", spec_.sizes);
    }
    if (spec_.rand_testmode == test_mode_t::SCALES_ATTR) {
        resampling.set_attr<std::vector<int64_t>>("sizes", {});
        resampling.set_attr<std::vector<float>>("scales", spec_.scales);
    }
    ops_.emplace_back(resampling);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t resampling_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.resampling.eltw_handler(*this, po_entry);
}

fill_status_t resampling_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.resampling.bin_handler(
            *this, spec_.data_format, po_entry);
}

fill_status_t resampling_graph_prb_t::handle_sum_() {
    return po_handler.resampling.sum_handler(*this);
}

void check_known_skipped_case_graph(
        const ::resampling::prb_t *prb, res_t *res) noexcept {
    ::resampling::check_known_skipped_case(prb, res);
    //Skip if source and destination datatypes are different.
    //Skip backward cases
    if (prb->sdt != prb->ddt || !(prb->dir & FLAG_FWD)) {
        res->state = SKIPPED, res->reason = KNOWN_LIMITATION;
        return;
    }
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return;
}

int doit(const ::resampling::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    resampling_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;
    const auto ins = par.get_in_ports();
    const auto outs_p = par.get_out_ports();

    auto cp = compile_partition(
            ::resampling::init_pd, prb, res, par, ins, outs_p);
    const auto cp_dst_lt = cp.query_logical_tensor(outs_p[0].get_id());

    dnnl::graph::engine &eng = get_test_engine();
    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(cp_dst_lt, dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->tag);
    auto dst_dt = make_dnn_mem(cp_dst_lt, prb->tag);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    // When post-ops occur, the relative difference can change
    // between the output from reference and the kernel. The compare
    // function usually uses to compare a relative difference.
    // Therefore, we should not lead to a situation where the
    // relative difference is very small after executing a
    // post-ops operation. Therefore, all values for binary post_ops
    // are positive when the linear algorithm is present. This is
    // important because there may be small differences in the result
    // between the expected value and the gotten value with this algorithm.
    const bool only_positive_values = prb->alg == ::resampling::linear;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), prb->tag));
        const int idx = 0;
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx),
                binary_po_dt.back(), binary_po_fp.back(), only_positive_values);
    }

    compare::compare_t cmp;
    const bool operations_order_can_be_different
            = prb->alg == ::resampling::linear;
    if (operations_order_can_be_different)
        ::resampling::add_additional_check_to_compare(cmp);

    //TODO: add for backward.
    if (prb->dir & FLAG_FWD) {
        SAFE(::resampling::fill_src(prb, src_dt, src_fp, res), WARN);
        dnnl::graph::tensor src_tensor(
                ins[0], eng, static_cast<void *>(src_dt));
        dnnl::graph::tensor dst_tensor(
                cp_dst_lt, eng, static_cast<void *>(dst_dt));

        std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
        dnnl::graph::tensor sizes_tensor;
        dnnl::graph::tensor bin_tensor;
        dnnl::graph::tensor sum_src1_tensor;
        std::vector<int64_t> sizes_v(spec.sizes);
        if (spec.rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
            sizes_tensor = dnnl::graph::tensor(
                    ins[1], eng, static_cast<void *>(sizes_v.data()));
            tensors_in.emplace_back(sizes_tensor);
        }
        if (graph_prb.has_post_bin()) {
            bin_tensor = dnnl::graph::tensor(
                    ins.back(), eng, static_cast<void *>(binary_po_dt.back()));
            tensors_in.emplace_back(bin_tensor);
        } else if (graph_prb.has_post_sum()) {
            sum_src1_tensor = dnnl::graph::tensor(
                    ins.back(), eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        }
        std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

        SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

        if (is_bench_mode(CORR)) {
            compute_ref_fwd(prb, src_fp, dst_fp, binary_po_fp);
            const float linear_trh = epsilon_dt(prb->sdt) > epsilon_dt(prb->ddt)
                    ? epsilon_dt(prb->sdt) // conversion error sdt->ddt
                    : 7 * epsilon_dt(prb->ddt); // algorithm calculation error
            float trh = prb->alg == ::resampling::nearest ? 0.f : linear_trh;

            cmp.set_threshold(trh);
            // No sense to test zero trust for upsampling since it produces
            // valid zeros.
            // TODO: validate this once again.
            cmp.set_zero_trust_percent(100.f);
            SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
        }
        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
    }
    return OK;
}

} // namespace resampling
} // namespace benchdnnext
