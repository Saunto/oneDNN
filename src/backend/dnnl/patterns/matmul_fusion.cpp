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

#include "utils/pm/pbuilder.hpp"

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
 * \brief This provides matmul-related fusion, i.e.
 *        matmul-relu fusion
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(matmul_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, matmul_post_ops_chain_fusion)
        .set_priority(8.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<2>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pmatmul, 0)}, "popt");

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

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardTanh,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh, impl::op_kind::Add,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {swish_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::matmul_post_ops_chain_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, matmul_bias_post_ops_chain_fusion)
        .set_priority(8.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<2>);
                    pm::pb_op *biasadd
                            = pgraph->append_op(impl::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, pmatmul, 0)});

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, biasadd, 0)}, "popt");

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

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardTanh,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh, impl::op_kind::Add,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {swish_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul);
                    pmatmul->append_decision_function(check_input_num<3>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pmatmul, 0)}, "popt");

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

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::Exp,
                                    impl::op_kind::GELU,
                                    impl::op_kind::HardTanh,
                                    impl::op_kind::HardSwish,
                                    impl::op_kind::Log, impl::op_kind::Sigmoid,
                                    impl::op_kind::SoftPlus, impl::op_kind::Pow,
                                    impl::op_kind::ReLU, impl::op_kind::Round,
                                    impl::op_kind::Sqrt, impl::op_kind::Square,
                                    impl::op_kind::Tanh, impl::op_kind::Add,
                                    impl::op_kind::Multiply,
                                    impl::op_kind::Maximum,
                                    impl::op_kind::Minimum,
                                    impl::op_kind::Divide,
                                    impl::op_kind::Subtract},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {swish_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    pgraph->append_repetition(alt_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::matmul_bias_post_ops_chain_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_div_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *matmul = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequant_data, 0),
                                    in_edge(1, dequant_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, div, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_matmul_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
              matmul
                |
              [bias]*                     [dequant_add]
                |                             /
        [ Abs/Clamp/Elu/Exp/GELU/HardTanh/HardSwish/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/ Add*[0,1] ]*[0,3]
                |
            [quant_out]*  
                |      
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_matmul_post_ops_fusion)
        .set_priority(9.9f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data = pgraph->append_op(
                            impl::op_kind::Dequantize, "dequant_data");

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");

                    pm::pb_op *pmatmul
                            = pgraph->append_op(impl::op_kind::MatMul,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "matmul");

                    // Optional bias_add
                    auto popt_bias_graph
                            = std::make_shared<pb_graph_t>("poptional_bias");
                    pm::pb_op *pbias = popt_bias_graph->append_op(
                            impl::op_kind::BiasAdd, "pbias");
                    pbias->append_decision_function(
                            check_producer_input_num<2>);
                    popt_bias_graph->create_input_port(0, pbias, 0);
                    popt_bias_graph->create_output_port(0, pbias, 0);
                    auto popt_bias = pgraph->append_optional(popt_bias_graph,
                            in_edges_t {in_edge(0, pmatmul, 0)}, "popt_bias");

                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op *padd = padd_graph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    padd->append_decision_function(
                            check_post_ops_only_one_add<impl::op_kind::MatMul>);
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 0);
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
                    pm::pb_op *pop = peltwise_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Divide, impl::op_kind::Elu,
                                    impl::op_kind::Exp, impl::op_kind::GELU,
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
                    pm::pb_op *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_matmul_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_bf16_matmul_div_add_fusion)
        .set_priority(10.5f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *dequant_weight
                            = pgraph->append_op(impl::op_kind::Dequantize);
                    pm::pb_op *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op *matmul = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, typecast_data, 0),
                                    in_edge(1, typecast_weight, 0)});
                    matmul->append_decision_function(check_input_num<2>);

                    pm::pb_op *div = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul, 0)});
                    pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, div, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_matmul_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        |                  |
   typecast_data    typecast_weight
        \_____       _____/
              matmul
                |
              [bias]*    [dequant_other -> typecast_other]* for Add
                |          /
 [ ReLU/GELU/Divide/Multiply/Add ]
                |
  [typecast_out -> quant_out]*
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_bf16_matmul_post_ops_fusion)
        .set_priority(10.4f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *dequant_data
                            = pgraph->append_op(impl::op_kind::Dequantize);

                    // Optional quant_weight
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_quant_weight");
                    pm::pb_op *pquant = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant");
                    popt_graph->create_input_port(0, pquant, 0);
                    popt_graph->create_output_port(0, pquant, 0);
                    auto popt = pgraph->append_optional(popt_graph, "popt");

                    pm::pb_op *dequant_weight = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, popt, 0)}, "dequant_weight");
                    pm::pb_op *typecast_data
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_data, 0)});
                    typecast_data->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op *typecast_weight
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequant_weight, 0)});
                    typecast_weight->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);

                    pm::pb_op *matmul = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, typecast_data, 0),
                                    in_edge(1, typecast_weight, 0)});

                    // Optional bias_add
                    auto popt_bias_graph
                            = std::make_shared<pb_graph_t>("poptional_bias");
                    pm::pb_op *typecast_bias = popt_bias_graph->append_op(
                            impl::op_kind::TypeCast, "tc_bias");
                    typecast_bias->append_decision_function(
                            check_output_dtype<impl::data_type::bf16>);
                    pm::pb_op *pbias = popt_bias_graph->append_op(
                            impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(1, typecast_bias, 0)}, "pbias");
                    pbias->append_decision_function(
                            check_producer_input_num<2>);
                    popt_bias_graph->create_input_port(0, pbias, 0);
                    popt_bias_graph->create_output_port(0, pbias, 0);
                    auto popt_bias = pgraph->append_optional(popt_bias_graph,
                            in_edges_t {in_edge(0, matmul, 0)}, "popt_bias");

                    // post add with dequant->typecast
                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op *typecast_add
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, pdequant_add, 0)});
                    pm::pb_op *padd = padd_graph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(1, typecast_add, 0)}, "padd");
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 0);
                    padd_graph->create_output_port(0, padd, 0);

                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Divide, impl::op_kind::GELU,
                                    impl::op_kind::Add},
                            "pother_postop");
                    other_postop_graph->create_input_port(0, pop, 0);
                    other_postop_graph->create_input_port(1, pop, 1);
                    other_postop_graph->create_output_port(0, pop, 0);

                    auto alt_graph = std::make_shared<pb_graph_t>("alt_graph");
                    auto palt = alt_graph->append_alternation(
                            {padd_graph, other_postop_graph}, "palt");
                    alt_graph->create_input_port(0, palt, 0);
                    alt_graph->create_output_port(0, palt, 0);

                    auto prep = pgraph->append_optional(alt_graph,
                            in_edges_t {in_edge(0, popt_bias, 0)},
                            "prepetition");

                    // Optional typecast_out + quant_out
                    auto popt_qout_graph = std::make_shared<pb_graph_t>(
                            "poptional_tc_quant_out");
                    pm::pb_op *ptc_out = popt_qout_graph->append_op(
                            impl::op_kind::TypeCast, "ptc_out");
                    pm::pb_op *pquant_out = popt_qout_graph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, ptc_out, 0)}, "pquant_out");
                    popt_qout_graph->create_input_port(0, ptc_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)},
                            "popt_tc_quant_out");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_matmul_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_MHA_fusion)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");
                    auto quantize_query
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, query_transpose, 0)},
                                    "quantize_query");
                    auto dequantize_query
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_query, 0)},
                                    "dequantize_query");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");
                    auto key_transpose2
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_transpose, 0)},
                                    "key_transpose2");
                    auto quantize_key
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, key_transpose2, 0)},
                                    "quantize_key");
                    auto dequantize_key
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_key, 0)},
                                    "dequantize_key");
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_query, 0),
                                    in_edge(1, dequantize_key, 0)},
                            "matmul_qk");

                    auto fscore_scale = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");
                    auto quantize_softmax
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, softmax, 0)},
                                    "quantize_softmax");
                    auto dequantize_softmax = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)},
                            "dequantize_softmax");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");
                    auto quantize_value
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, value_transpose, 0)},
                                    "quantize_value");
                    auto dequantize_value
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_value, 0)},
                                    "dequantize_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, dequantize_softmax, 0),
                                    in_edge(1, dequantize_value, 0)},
                            "matmul_v");
                    pgraph->append_op(impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, matmul_v, 0)},
                            "transpose_output");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, f32_MHA_fusion)
        .set_priority(20.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");

                    // Optional key_transpose
                    auto popt_graph = std::make_shared<pb_graph_t>(
                            "poptional_transpose");
                    auto ptranspose = popt_graph->append_op(
                            impl::op_kind::StaticTranspose, "pkey_transpose");
                    popt_graph->create_input_port(0, ptranspose, 0);
                    popt_graph->create_output_port(0, ptranspose, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, key_transpose, 0)}, "popt");

                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, query_transpose, 0),
                                    in_edge(1, popt, 0)},
                            "matmul_qk");

                    // Optional fscore_scale
                    auto popt_graph2
                            = std::make_shared<pb_graph_t>("poptional_scale");
                    auto pfscore_scale = popt_graph2->append_op(
                            impl::op_kind::Divide, "pfscore_scale");
                    popt_graph2->create_input_port(0, pfscore_scale, 0);
                    popt_graph2->create_output_port(0, pfscore_scale, 0);
                    auto popt2 = pgraph->append_optional(
                            popt_graph2, {in_edge(0, matmul_qk, 0)}, "popt2");

                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, popt2, 0)}, "fscore_add");

                    // Optional Pre Reshape of SoftMax
                    auto popt_graph3
                            = std::make_shared<pb_graph_t>("poptional_reshape");
                    auto pre_reshape = popt_graph3->append_op(
                            impl::op_kind::StaticReshape, "pre_reshape");
                    popt_graph3->create_input_port(0, pre_reshape, 0);
                    popt_graph3->create_output_port(0, pre_reshape, 0);
                    auto popt3 = pgraph->append_optional(
                            popt_graph3, {in_edge(0, fscore_add, 0)}, "popt3");

                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, popt3, 0)}, "softmax");

                    // Optional Post Reshape of SoftMax
                    auto popt_graph4
                            = std::make_shared<pb_graph_t>("poptional_reshape");
                    auto post_reshape = popt_graph4->append_op(
                            impl::op_kind::StaticReshape, "post_reshape");
                    popt_graph4->create_input_port(0, post_reshape, 0);
                    popt_graph4->create_output_port(0, post_reshape, 0);
                    auto popt4 = pgraph->append_optional(
                            popt_graph4, {in_edge(0, softmax, 0)}, "popt3");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");

                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, popt4, 0),
                                    in_edge(1, value_transpose, 0)},
                            "matmul_v");
                    auto post_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, matmul_v, 0)},
                                    "transpose_output");

                    // Optional Reshape
                    auto popt_reshape_graph = std::make_shared<pb_graph_t>(
                            "poptional_reshape_out");
                    pm::pb_op *preshape_out = popt_reshape_graph->append_op(
                            impl::op_kind::StaticReshape, "preshape_out");
                    popt_reshape_graph->create_input_port(0, preshape_out, 0);
                    popt_reshape_graph->create_output_port(0, preshape_out, 0);
                    pgraph->append_optional(popt_reshape_graph,
                            in_edges_t {in_edge(0, post_transpose, 0)},
                            "popt_reshape_out");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::f32_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_bf16_MHA_fusion)
        .set_priority(5.0f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto query_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "query_reshape");
                    auto query_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, query_reshape, 0)},
                                    "query_transpose");
                    auto bf16_to_f32_query
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, query_transpose, 0)},
                                    "bf16_to_f32_query");
                    auto quantize_query = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_query, 0)},
                            "quantize_query");
                    auto dequantize_query
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_query, 0)},
                                    "dequantize_query");
                    auto f32_to_bf16_query = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_query, 0)},
                            "f32_to_bf16_query");

                    auto key_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "key_reshape");
                    auto key_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_reshape, 0)},
                                    "key_transpose");
                    auto key_transpose2
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, key_transpose, 0)},
                                    "key_transpose2");
                    auto bf16_to_f32_key
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, key_transpose2, 0)},
                                    "bf16_to_f32_key");
                    auto quantize_key
                            = pgraph->append_op(impl::op_kind::Quantize,
                                    in_edges_t {in_edge(0, bf16_to_f32_key, 0)},
                                    "quantize_key");
                    auto dequantize_key
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_key, 0)},
                                    "dequantize_key");
                    auto f32_to_bf16_key
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, dequantize_key, 0)},
                                    "f32_to_bf16_key");
                    auto matmul_qk = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, f32_to_bf16_query, 0),
                                    in_edge(1, f32_to_bf16_key, 0)},
                            "matmul_qk");

                    auto fscore_scale = pgraph->append_op(impl::op_kind::Divide,
                            in_edges_t {in_edge(0, matmul_qk, 0)},
                            "fscore_scale");
                    auto fscore_add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, fscore_scale, 0)},
                            "fscore_add");
                    auto softmax = pgraph->append_op(impl::op_kind::SoftMax,
                            in_edges_t {in_edge(0, fscore_add, 0)}, "softmax");
                    auto bf16_to_f32_softmax
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, softmax, 0)},
                                    "bf16_to_f32_softmax");
                    auto quantize_softmax = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_softmax, 0)},
                            "quantize_softmax");
                    auto dequantize_softmax = pgraph->append_op(
                            impl::op_kind::Dequantize,
                            in_edges_t {in_edge(0, quantize_softmax, 0)},
                            "dequantize_softmax");
                    auto f32_to_bf16_softmax = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_softmax, 0)},
                            "f32_to_bf16_softmax");

                    auto value_reshape = pgraph->append_op(
                            impl::op_kind::StaticReshape, "value_reshape");
                    auto value_transpose
                            = pgraph->append_op(impl::op_kind::StaticTranspose,
                                    in_edges_t {in_edge(0, value_reshape, 0)},
                                    "value_transpose");
                    auto bf16_to_f32_value
                            = pgraph->append_op(impl::op_kind::TypeCast,
                                    in_edges_t {in_edge(0, value_transpose, 0)},
                                    "bf16_to_f32_value");
                    auto quantize_value = pgraph->append_op(
                            impl::op_kind::Quantize,
                            in_edges_t {in_edge(0, bf16_to_f32_value, 0)},
                            "quantize_value");
                    auto dequantize_value
                            = pgraph->append_op(impl::op_kind::Dequantize,
                                    in_edges_t {in_edge(0, quantize_value, 0)},
                                    "dequantize_value");
                    auto f32_to_bf16_value = pgraph->append_op(
                            impl::op_kind::TypeCast,
                            in_edges_t {in_edge(0, dequantize_value, 0)},
                            "f32_to_bf16_value");
                    auto matmul_v = pgraph->append_op(impl::op_kind::MatMul,
                            in_edges_t {in_edge(0, f32_to_bf16_softmax, 0),
                                    in_edge(1, f32_to_bf16_value, 0)},
                            "matmul_v");
                    pgraph->append_op(impl::op_kind::StaticTranspose,
                            in_edges_t {in_edge(0, matmul_v, 0)},
                            "transpose_output");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::int8_MHA);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
