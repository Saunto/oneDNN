/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "backend/dnnl/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

#define MAX_NUM_OF_CONCAT 64

namespace {
bool check_scales_zps_all_equal(op_t *op) {
    auto out_port = op->get_output_value(0);
    if (out_port->get_consumers().empty()) return false;

    auto &out_op = out_port->get_consumers()[0].get_op();
    // We only want to accept int8 concat with inputs using
    // the same scales and zps. Concat does not change range
    // of values so output scales and zps should be same as well.
    if (!out_op.has_attr("scales") || !out_op.has_attr("zps")) return false;
    const auto expected_scales = out_op.get_attr<std::vector<float>>("scales");
    const auto expected_zps = out_op.get_attr<std::vector<int64_t>>("zps");

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        auto in_port = op->get_input_value(i);
        if (!in_port->has_producer()) return false;

        auto &in_op = in_port->get_producer();
        if (!in_op.has_attr("scales") || !in_op.has_attr("zps")) return false;
        auto scales = in_op.get_attr<std::vector<float>>("scales");
        auto zps = in_op.get_attr<std::vector<int64_t>>("zps");
        if (scales != expected_scales || zps != expected_zps) return false;
    }

    return true;
}
} // namespace

/*!
 * \brief This provides concat-related fusion, i.e.
 *        int8-concat fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(concat_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_concat_fusion)
        .set_priority(8.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    in_edges_t input_edges;
                    for (size_t i = 0; i < MAX_NUM_OF_CONCAT; ++i) {
                        pm::pb_op *dequant
                                = pgraph->append_op(impl::op_kind::Dequantize);
                        input_edges.emplace_back(in_edge(i, dequant, 0));
                    }
                    pm::pb_op *concat = pgraph->append_op(
                            impl::op_kind::Concat, input_edges);
                    concat->append_decision_function(
                            check_scales_zps_all_equal);

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, concat, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_concat_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
