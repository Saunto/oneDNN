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

namespace {
template <bool GROUPED>
bool check_grouped(op_t *op) {
    if (GROUPED) {
        return op->has_attr("groups") && op->get_attr<int64_t>("groups") > 1;
    } else {
        return !op->has_attr("groups") || op->get_attr<int64_t>("groups") <= 1;
    }
}

// Block creators used to construct large patterns
pm::pb_op *conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    return conv_bias_dst;
};

pm::pb_op *conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, bool grouped = false, bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }

    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return relu;
};

pm::pb_op *conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, pm::pb_op *post_src, bool grouped = false,
        bool use_biasadd = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution, in_edges);
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);

    in_edges_t add_in_edges = in_edges_t {in_edge(0, conv_bias_dst, 0)};
    if (post_src) { add_in_edges.emplace_back(in_edge(1, post_src, 0)); }
    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add, add_in_edges);
    add->allow_internal_inputs({0, 1});

    pm::pb_op *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    return relu;
};

pm::pb_op *int8_conv_bias(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op *quant_dst = pgraph->append_op(
            impl::op_kind::Quantize, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    return quant_dst;
};

pm::pb_op *int8_conv_bias_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    in_edges_t in_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    pm::pb_op *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, conv_bias_dst, 0)});
    pm::pb_op *quant_dst = pgraph->append_op(
            impl::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
    return quant_dst;
};

pm::pb_op *int8_conv_bias_add_relu(const std::shared_ptr<pb_graph_t> &pgraph,
        pm::pb_op *input, pm::pb_op *post_src, bool grouped = false,
        bool use_biasadd = false, bool use_quant_wei = false,
        bool f32_output = false) {
    in_edges_t in_edges, post_src_edges;
    if (input) { in_edges = in_edges_t {in_edge(0, input, 0)}; }
    if (post_src) { post_src_edges = in_edges_t {in_edge(0, post_src, 0)}; }
    pm::pb_op *dequant_src
            = pgraph->append_op(impl::op_kind::Dequantize, in_edges);
    pm::pb_op *dequant_wei = nullptr;
    if (use_quant_wei) {
        pm::pb_op *quant_wei = pgraph->append_op(impl::op_kind::Quantize);
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize,
                in_edges_t {in_edge(0, quant_wei, 0)});
    } else {
        dequant_wei = pgraph->append_op(impl::op_kind::Dequantize);
    }
    pm::pb_op *dequant_other
            = pgraph->append_op(impl::op_kind::Dequantize, post_src_edges);
    pm::pb_op *conv = pgraph->append_op(impl::op_kind::Convolution,
            in_edges_t {
                    in_edge(0, dequant_src, 0), in_edge(1, dequant_wei, 0)});
    pm::pb_op *conv_bias_dst = nullptr;
    if (use_biasadd) {
        conv->append_decision_function(check_input_num<2>);
        pm::pb_op *biasadd = pgraph->append_op(
                impl::op_kind::BiasAdd, in_edges_t {in_edge(0, conv, 0)});
        conv_bias_dst = biasadd;
    } else {
        conv->append_decision_function(check_input_num<3>);
        conv_bias_dst = conv;
    }
    conv->append_decision_function(
            grouped ? check_grouped<true> : check_grouped<false>);
    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
            in_edges_t {in_edge(0, conv_bias_dst, 0),
                    in_edge(1, dequant_other, 0)});
    add->allow_internal_inputs({0, 1});
    pm::pb_op *relu = pgraph->append_op(
            impl::op_kind::ReLU, in_edges_t {in_edge(0, add, 0)});
    if (f32_output) {
        return relu;
    } else {
        pm::pb_op *quant_dst = pgraph->append_op(
                impl::op_kind::Quantize, in_edges_t {in_edge(0, relu, 0)});
        return quant_dst;
    }
};

// The F(x)+x basic residual block
pm::pb_op *int8_identical_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op *quant_dst1 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, input, grouped, use_biasadd);
    return quant_dst1;
};

// The F(x)+G(x) basic residual block
pm::pb_op *int8_convolutional_basic_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op *quant_dst0
            = int8_conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op *quant_dst1 = int8_conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op *quant_dst2 = int8_conv_bias_add_relu(
            pgraph, quant_dst0, quant_dst1, grouped, use_biasadd);
    return quant_dst2;
};

// The F(x)+x bottleneck residual block
pm::pb_op *int8_identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false, bool f32_output = false) {
    pm::pb_op *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst2 = int8_conv_bias_add_relu(pgraph, quant_dst1, input,
            grouped, use_biasadd, use_quant_wei, f32_output);
    return quant_dst2;
};

// The F(x)+G(x) bottleneck residual block
pm::pb_op *int8_convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst2 = int8_conv_bias(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst3 = int8_conv_bias_add_relu(pgraph, quant_dst1,
            quant_dst2, grouped, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op *int8_convolutional_bottleneck_resblock_v2(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false,
        bool use_quant_wei = false) {
    pm::pb_op *quant_dst0 = int8_conv_bias_relu(
            pgraph, input, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst1 = int8_conv_bias_relu(
            pgraph, quant_dst0, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst2 = int8_conv_bias(
            pgraph, quant_dst1, grouped, use_biasadd, use_quant_wei);
    pm::pb_op *quant_dst3 = int8_conv_bias_add_relu(
            pgraph, input, quant_dst2, grouped, use_biasadd, use_quant_wei);
    return quant_dst3;
};

pm::pb_op *convolutional_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op *dst2 = conv_bias(pgraph, input, grouped, use_biasadd);
    pm::pb_op *dst3
            = conv_bias_add_relu(pgraph, dst1, dst2, grouped, use_biasadd);
    return dst3;
};

pm::pb_op *identical_bottleneck_resblock(
        const std::shared_ptr<pb_graph_t> &pgraph, pm::pb_op *input,
        bool grouped = false, bool use_biasadd = false) {
    pm::pb_op *dst0 = conv_bias_relu(pgraph, input, grouped, use_biasadd);
    pm::pb_op *dst1 = conv_bias_relu(pgraph, dst0, grouped, use_biasadd);
    pm::pb_op *dst2
            = conv_bias_add_relu(pgraph, dst1, input, grouped, use_biasadd);
    return dst2;
};

} // namespace

/*!
 * \brief This provides conv-related fusion, i.e.
 *        conv-relu fusion, conv-bn fusion, conv-sum fusion, conv-bn-sum fusion, etc.
 *        The process includes follow steps:
 *          1. look for fusion pattern on the graph
 *          2. If found, verify if this transformation is safe / correct
 *          3. replace the pattern with a fused op, update the graph
 */
DNNL_BACKEND_REGISTER_PASSES_DEF_BEGIN(conv_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_depthwise_fusion)
        .set_priority(10.2f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *depthwise
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, conv, 0)});
                    depthwise->append_decision_function(check_input_num<2>);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::dnnl_conv_depthwise);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

/*
                    [quant_weight]*
        |                  |
   dequant_data     dequant_weight
        \_____       _____/
               conv
                |
              [bias]*                     [dequant_add]
                |                             /
        [ Abs/Clamp/Elu/GELU/HardTanh/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/ Add*[0,1] ]*[0,3]
                |
            [quant_out]*  
                |      
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_conv_post_ops_fusion)
        .set_priority(10.5f)
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

                    pm::pb_op *pconv
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, dequant_data, 0),
                                            in_edge(1, dequant_weight, 0)},
                                    "conv");

                    // Optional bias_add
                    auto popt_bias_graph
                            = std::make_shared<pb_graph_t>("poptional_bias");
                    pm::pb_op *pbias = popt_graph->append_op(
                            impl::op_kind::BiasAdd, "pbias");
                    pbias->append_decision_function(
                            check_producer_input_num<2>);
                    popt_bias_graph->create_input_port(0, pbias, 0);
                    popt_bias_graph->create_output_port(0, pbias, 0);
                    auto popt_bias = pgraph->append_optional(popt_bias_graph,
                            in_edges_t {in_edge(0, pconv, 0)}, "popt_bias");

                    auto padd_graph
                            = std::make_shared<pb_graph_t>("padd_graph");
                    pm::pb_op *pdequant_add = padd_graph->append_op(
                            impl::op_kind::Dequantize, "dequant_add");
                    pm::pb_op *padd = padd_graph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(1, pdequant_add, 0)}, "padd");
                    padd->append_decision_function(check_post_ops_only_one_add<
                            impl::op_kind::Convolution>);
                    padd_graph->create_input_port(0, padd, 0);
                    padd_graph->create_input_port(1, pdequant_add, 1);
                    padd_graph->create_output_port(0, padd, 0);

                    auto peltwise_graph
                            = std::make_shared<pb_graph_t>("peltwise_graph");
                    pm::pb_op *pop = peltwise_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::Sigmoid,
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
                            {padd_graph, peltwise_graph}, "palternation");
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
                    pm::pb_op *pquant_out = popt_graph->append_op(
                            impl::op_kind::Quantize, "pquant_out");
                    popt_qout_graph->create_input_port(0, pquant_out, 0);
                    popt_qout_graph->create_output_port(0, pquant_out, 0);
                    pgraph->append_optional(popt_qout_graph,
                            in_edges_t {in_edge(0, prep, 0)}, "popt_quant_out");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_swish_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);

                    pm::pb_op *bias = pgraph->append_op(impl::op_kind::BiasAdd,
                            in_edges_t {in_edge(0, conv, 0)});

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, bias, 0)});

                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, bias, 0),
                                    in_edge(1, sigmoid, 0)});
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv = pgraph->append_op(
                            impl::op_kind::Convolution, "p-conv");
                    conv->append_decision_function(check_input_num<3>);

                    pm::pb_op *sigmoid
                            = pgraph->append_op(impl::op_kind::Sigmoid,
                                    in_edges_t {in_edge(0, conv, 0)}, "p-sig");

                    pgraph->append_op(impl::op_kind::Multiply,
                            in_edges_t {in_edge(0, conv, 0),
                                    in_edge(1, sigmoid, 0)},
                            "p-mul");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op
                            = std::make_shared<op_t>(op_kind::conv_bias_swish);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_simple_resblock_fusion)
        .set_priority(5.f) // low priority to avoid current functionality
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv0
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv0->append_decision_function(check_input_num<2>);
                    pm::pb_op *relu0 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv0, 0)});

                    pm::pb_op *conv1
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu0, 0)});
                    conv1->append_decision_function(check_input_num<2>);
                    pm::pb_op *relu1 = pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, conv1, 0)});

                    pm::pb_op *conv2
                            = pgraph->append_op(impl::op_kind::Convolution,
                                    in_edges_t {in_edge(0, relu1, 0)});
                    conv2->append_decision_function(check_input_num<2>);

                    pm::pb_op *add = pgraph->append_op(impl::op_kind::Add,
                            in_edges_t {in_edge(0, conv2, 0)});
                    add->allow_internal_inputs({0, 1});

                    pgraph->append_op(impl::op_kind::ReLU,
                            in_edges_t {in_edge(0, add, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_simple_resblock);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

// Two conv fusion for f32 resnet
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, conv_bias_relu_conv_bias_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *relu0 = conv_bias_relu(pgraph, nullptr);
                    conv_bias_relu(pgraph, relu0);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, conv_bias_relu_conv_bias_add_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *relu0 = conv_bias_relu(pgraph, nullptr);
                    conv_bias_add_relu(pgraph, relu0, nullptr);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, identical_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    identical_bottleneck_resblock(pgraph, nullptr, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, convolutional_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

// Two conv fusion for int8 resnet
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_conv_bias_relu_conv_bias_relu_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *quant_dst0
                            = int8_conv_bias_relu(pgraph, nullptr);
                    int8_conv_bias_relu(pgraph, quant_dst0);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, int8_identical_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_identical_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_identical_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl,
        int8_convolutional_bottleneck_resblock_fusion)
        .set_priority(5.f) // increase the priority if needed
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_convolutional_bottleneck_resblock(pgraph, nullptr);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                    output = int8_identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet34_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_identical_basic_resblock(pgraph, nullptr);
                    output = int8_identical_basic_resblock(pgraph, output);
                    output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet34_stage_2_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, int8_resnet34_stage_3_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_basic_resblock(pgraph, nullptr);
                    // 5 F(x)+x blocks
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_basic_resblock(pgraph, output);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, f32_resnet50_stage_1_4_fusion)
        .set_priority(22.f) // high priority to support itex
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    output = identical_bottleneck_resblock(pgraph, output);
                    output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                    output = identical_bottleneck_resblock(
                            pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, f32_resnet50_stage_2_fusion)
        .set_priority(22.1f) // high priority to support itex
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, f32_resnet50_stage_3_fusion)
        .set_priority(22.2f) // high priority to support itex
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(pgraph, nullptr);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(pgraph, output);
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = convolutional_bottleneck_resblock(
                            pgraph, nullptr, false, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = identical_bottleneck_resblock(
                                pgraph, output, false, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::float_conv_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, itex_int8_resnet50_stage_1_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, itex_int8_resnet50_stage_2_fusion)
        .set_priority(22.2f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    // 3 F(x)+x blocks
                    const size_t identical_residual_block_num = 3;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

// For itex int8 rn50 only (include the weight quantize into pattern)
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, itex_int8_resnet50_stage_3_fusion)
        .set_priority(22.3f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    const size_t identical_residual_block_num = 5;
                    for (size_t i = 0; i < identical_residual_block_num; i++)
                        output = int8_identical_bottleneck_resblock(
                                pgraph, output, false, true, true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, itex_int8_resnet50_stage_4_fusion)
        .set_priority(22.1f) // high priority to support lz models
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *output = nullptr;
                    output = int8_convolutional_bottleneck_resblock_v2(
                            pgraph, nullptr, false, true, true);
                    output = int8_identical_bottleneck_resblock(
                            pgraph, output, false, true, true);
                    output = int8_identical_bottleneck_resblock(pgraph, output,
                            false, true, true, /* f32 output */ true);
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::int8_conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

/*
              \   /
              conv
                |
        [ Abs/Clamp/Elu/GELU/HardTanh/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |       
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_post_ops_fusion)
        .set_priority(9.7f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *pconv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    pconv->append_decision_function(check_input_num<2>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, pconv, 0)}, "popt");

                    // TODO(Yixin): special post op handle: swish is composed
                    // by sigmoid and multiply
                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::Sigmoid,
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

                    pgraph->append_repetition(other_postop_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

/*
              \   /
              conv
                |
              bias
                |
        [ Abs/Clamp/Elu/GELU/HardTanh/Log/Sigmoid/SoftPlus/
          Pow/ReLU/Round/Sqrt/Square/Tanh/Add/Multiply/
          Maximum/Minimum/Divide/Subtract]*[0,3] 
                |      
*/
DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(dnnl, conv_bias_post_ops_fusion)
        .set_priority(9.8f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<2>);
                    pm::pb_op *biasadd
                            = pgraph->append_op(impl::op_kind::BiasAdd,
                                    in_edges_t {in_edge(0, conv, 0)});

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, biasadd, 0)}, "popt");

                    // TODO(Yixin): special post op handle: swish is composed
                    // by sigmoid and multiply
                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::Sigmoid,
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

                    pgraph->append_repetition(other_postop_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *conv
                            = pgraph->append_op(impl::op_kind::Convolution);
                    conv->append_decision_function(check_input_num<3>);

                    // Optional BN
                    auto popt_graph
                            = std::make_shared<pb_graph_t>("poptional_bn");
                    auto pbn = popt_graph->append_op(
                            impl::op_kind::BatchNormInference, "pbn");
                    popt_graph->create_input_port(0, pbn, 0);
                    popt_graph->create_output_port(0, pbn, 0);
                    auto popt = pgraph->append_optional(
                            popt_graph, {in_edge(0, conv, 0)}, "popt");

                    // TODO(Yixin): special post op handle: swish is composed
                    // by sigmoid and multiply
                    auto other_postop_graph = std::make_shared<pb_graph_t>(
                            "pother_postop_graph");
                    pm::pb_op *pop = other_postop_graph->append_alternation(
                            {impl::op_kind::Abs, impl::op_kind::Clamp,
                                    impl::op_kind::Elu, impl::op_kind::GELU,
                                    impl::op_kind::HardTanh, impl::op_kind::Log,
                                    impl::op_kind::Sigmoid,
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

                    pgraph->append_repetition(other_postop_graph, {0, 0}, 0,
                            MAX_REPETITION, in_edges_t {in_edge(0, popt, 0)},
                            "prepetition");
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::conv_bias_post_ops_fusion);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PASS(
        dnnl, conv_bwd_weights_bwd_bias_fusion)
        .set_enable(false)
        .set_priority(9.7f)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    pm::pb_op *wildcard = pgraph->append_op(
                            impl::op_kind::Wildcard, "pwild");
                    pgraph->append_op(impl::op_kind::ConvolutionBackpropFilters,
                            in_edges_t {in_edge(1, wildcard, 0)});
                    pgraph->append_op(impl::op_kind::BiasAddBackprop,
                            in_edges_t {in_edge(0, wildcard, 0)});
                })
        .set_attr<FCreateV2FusedOp>(
                "FCreateV2FusedOp", []() -> std::shared_ptr<op_t> {
                    std::shared_ptr<op_t> fused_op = std::make_shared<op_t>(
                            op_kind::dnnl_conv_bwd_weights);
                    fused_op->set_attr<std::string>("backend", "dnnl");
                    return fused_op;
                });

DNNL_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
