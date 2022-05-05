/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/*!
 * \brief This provides convtranspose-related fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(convtranspose_fusion)

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
         \              /
           convtranspose
                |
              [bias]*                     [dequant_add]
                |                             /
        [ Abs/Clamp/Elu/GELU/HardTanh/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/ Add*[0,1] ]*[0,3]
                |
            [quant_out]*  
                |      
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_convtranspose_post_ops_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op_t *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op_t *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op_t *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op_t *pconvtranspose
                            = pgraph->append_op(impl::op_kind::ConvTranspose,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias_graph
                            = std::make_shared<pb_graph_t>("poptional_bias");
                    pm::pb_op_t *pbias = popt_graph->append_op(
                            impl::op_kind::BiasAdd, "pbias");
                    pbias->append_decision_function(
                            check_producer_input_num<2>);
                    popt_bias_graph->create_input_port(0, pbias, 0);
                    popt_bias_graph->create_output_port(0, pbias, 0);
                    auto popt_bias = pgraph->append_optional(popt_bias_graph,
                            in_edges_t {in_edge(0, pconvtranspose, 0)},
                            "popt_bias");

                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op_t *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op_t *padd = padd_graph->append_op(
                            impl::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    padd->append_decision_function(check_post_ops_only_one_add<
                            impl::op_kind::ConvTranspose>);
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 1);
                    padd_graph->create_output_port(0, padd, 0);

                    // special post op handle: swish is composed
                    // by sigmoid and multiply
                    auto swish_graph
                            = std::make_shared<pb_graph_t>("swish_graph");
                    auto psigmoid = swish_graph->append_op(
                            impl::op_kind::Sigmoid, "psigmoid");
                    auto pmultiply
                            = swish_graph->append_op(impl::op_kind::Multiply,
                                    {in_edge(0, psigmoid, 0)}, "pmultiply");
                    swish_graph->create_input_port(0, psigmoid, 0);
                    swish_graph->create_input_port(0, pmultiply, 1);
                    swish_graph->create_output_port(0, pmultiply, 0);

                    auto peltwise_graph
                            = std::make_shared<pb_graph_t>("peltwise_graph");
                    pm::pb_op_t *pop = peltwise_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardTanh,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh},
                            "peltwise");
                    peltwise_graph->create_input_port(0, pop, 0);
                    peltwise_graph->create_input_port(1, pop, 1);
                    peltwise_graph->create_output_port(0, pop, 0);

                    auto prep_graph
                            = std::make_shared<pb_graph_t>("prep_graph");
                    auto palt = prep_graph->append_alternation(
                            {padd_graph, swish_graph, peltwise_graph},
                            "palternation");
                    prep_graph->create_input_port(0, palt, 0);
                    prep_graph->create_input_port(1, palt, 1);
                    prep_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_repetition(prep_graph, {0, 0}, 0,
                            MAX_REPETITION,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_out");
                    pm::pb_op_t *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::quantized_convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, convtranspose_post_ops_fusion)
        .set_priority(10.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // convtranspose
                    auto convtranspose = pgraph->append_op(
                            impl::op_kind::ConvTranspose, "convtranspose");

                    // optional biasadd
                    auto biasadd_subgraph
                            = std::make_shared<pb_graph_t>("biasadd_subgraph");
                    auto biasadd = biasadd_subgraph->append_op(
                            impl::op_kind::BiasAdd, "biasadd");
                    biasadd->append_decision_function(
                            check_producer_input_num<2>);
                    biasadd_subgraph->create_input_port(0, biasadd, 0);
                    biasadd_subgraph->create_output_port(0, biasadd, 0);
                    auto optional_biasadd
                            = pgraph->append_optional(biasadd_subgraph,
                                    {in_edge(0, convtranspose, 0)}, "biasadd");

                    // special post op handle: swish is composed
                    // by sigmoid and multiply
                    auto swish_graph
                            = std::make_shared<pb_graph_t>("swish_graph");
                    auto psigmoid = swish_graph->append_op(
                            impl::op_kind::Sigmoid, "psigmoid");
                    auto pmultiply
                            = swish_graph->append_op(impl::op_kind::Multiply,
                                    {in_edge(0, psigmoid, 0)}, "pmultiply");
                    swish_graph->create_input_port(0, psigmoid, 0);
                    swish_graph->create_input_port(0, pmultiply, 1);
                    swish_graph->create_output_port(0, pmultiply, 0);

                    auto post_ops = std::make_shared<pb_graph_t>("post_ops");
                    auto alternation = post_ops->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Add,
                                    impl::op_kind::Clamp, impl::op_kind::Divide,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Multiply, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Subtract,
                                    impl::op_kind::Tanh},
                            "alternation");
                    post_ops->create_input_port(0, alternation, 0);
                    post_ops->create_output_port(0, alternation, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {swish_graph, post_ops}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto optional_post_ops = pgraph->append_optional(alt_graph,
                            in_edges_t {in_edge(0, optional_biasadd, 0)},
                            "optional_post_ops");
                    pgraph->create_input_port(0, convtranspose, 0);
                    pgraph->create_output_port(0, optional_post_ops, 0);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::convtranspose_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
