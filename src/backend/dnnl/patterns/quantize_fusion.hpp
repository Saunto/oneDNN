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
#ifndef BACKEND_DNNL_PATTERNS_QUANTIZE_FUSION_HPP
#define BACKEND_DNNL_PATTERNS_QUANTIZE_FUSION_HPP

#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/patterns/transformation_pattern.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

/*!
 * \brief This provides quantize fusion.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph = impl::utils::pm::pb_graph;
using FCreateV2FusedOp = impl::pass::FCreateV2FusedOp;
using FCreateV2Pattern = impl::pass::FCreateV2Pattern;

DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(quantize_fusion)

#define SET_BF16_CHECK() \
    append_decision_function([](op_t *graph_op) -> bool { \
        for (size_t i = 0; i < graph_op->num_inputs(); ++i) { \
            logical_tensor_t iport \
                    = graph_op->get_input_value(i)->get_logical_tensor(); \
            if (iport.data_type != impl::data_type::bf16) return false; \
        } \
        return true; \
    })

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, typecast_quantize_fusion)
        .set_priority(8.1f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](std::shared_ptr<pb_graph> pgraph) -> void {
                    pm::pb_op *typecast
                            = pgraph->append_op(impl::op_kind::TypeCast);
                    // check it is a bf16->f32 typecast
                    typecast->SET_BF16_CHECK();

                    pgraph->append_op(impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, typecast, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(impl::op_kind::Quantize);
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
