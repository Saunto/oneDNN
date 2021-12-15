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
#ifndef BACKEND_DNNL_PATTERNS_REORDER_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_REORDER_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = pm::pb_graph_t;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

/*!
 * \brief This provides reorder-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(reorder_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, reorder_sum_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *reorder = pgraph->append_op(
                            impl::op_kind::Reorder, "preorder");
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, reorder, 0)}, "padd");
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr("auto_broadcast")
                                || graph_op->get_attr<std::string>(
                                           "auto_broadcast")
                                == "none";
                    });
                    add->set_commutative_pair({0, 1});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::reorder_sum);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_reorder_fusion)
        .set_priority(10.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant = pgraph->append_op(
                            impl::op_kind::Dequantize, "pdequant");
                    pm::pb_op *reorder
                            = pgraph->append_op(impl::op_kind::Reorder,
                                    {in_edge(0, dequant, 0)}, "preorder");
                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, reorder, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_reorder);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_reorder_sum_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph> &pgraph) -> void {
                    pm::pb_op *dequant = pgraph->append_op(
                            impl::op_kind::Dequantize, "pdequant");
                    pm::pb_op *dequant_other = pgraph->append_op(
                            impl::op_kind::Dequantize, "pdequant_other");
                    pm::pb_op *reorder
                            = pgraph->append_op(impl::op_kind::Reorder,
                                    {in_edge(0, dequant, 0)}, "preorder");
                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            {in_edge(0, reorder, 0),
                                    in_edge(1, dequant_other, 0)},
                            "padd");
                    add->append_decision_function([](op_t *graph_op) -> bool {
                        return !graph_op->has_attr("auto_broadcast")
                                || graph_op->get_attr<std::string>(
                                           "auto_broadcast")
                                == "none";
                    });
                    add->set_commutative_pair({0, 1});
                    pgraph->append_op(impl::op_kind::Quantize,
                            {in_edge(0, add, 0)}, "pquant");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_reorder);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
