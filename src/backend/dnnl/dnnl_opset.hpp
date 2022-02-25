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

#ifndef BACKEND_DNNL_DNNL_OPSET_HPP
#define BACKEND_DNNL_DNNL_OPSET_HPP

#include <functional>

#include "interface/op_schema.hpp"

#include "backend/dnnl/dnnl_op_def.hpp"
#include "backend/dnnl/internal_ops.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

class dnnl_opset_t {
public:
    static void for_each_schema(const std::function<void(op_schema_t &&)> &fn) {
        // fusion ops
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_conv_depthwise, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        conv_bias_post_ops_chain_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        conv_post_ops_chain_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        convtranspose_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(bn_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_bias_bn, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_hardtanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_relu6, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_hardtanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_add_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        matmul_add_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(avgpool_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(maxpool_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_conv, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_conv_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_conv_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_conv_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_conv_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_conv_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_conv_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_conv_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        quantized_convtranspose_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_matmul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_bias_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_bias_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_matmul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_quant_wei_matmul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_relu_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8float_matmul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8float_matmul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_mul_scales, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_constant, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_add_zps, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_sub_zps, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(permute, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(to_group, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(expand, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(squeeze, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_maxpool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_maxpool_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_convolution, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_convtranspose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(relu_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(add_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(add_multiply, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(maximum_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        maximum_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(minimum_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        minimum_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(multiply_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        multiply_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_reduction, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        reduction_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8x8float_matmul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8float_matmul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_matmul_bias_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(maximum_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(minimum_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(multiply_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(x8s8f32_conv, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_conv_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_conv_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_conv_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_conv_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_conv_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_bias_sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_bias_gelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_matmul_bias_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv_bias, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv_bias_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8s8f32_quant_wei_conv_bias_add_relu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_avgpool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        int8_avgpool_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_pool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_bn_folding, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_conv_bwd_data, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_conv_bwd_weights, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8x8float_matmul_div, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        x8x8float_matmul_div_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_batchnorm, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_batchnorm_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_binary, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(reorder_sum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(int8_reorder, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_eltwise, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_shuffle, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        interpolate_fusion, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_sum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_prelu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_div, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(matmul_div_add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_softmax_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_logsoftmax_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_resampling, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_resampling_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_concat, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_layernorm_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_pool_bwd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_matmul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        dnnl_logsoftmax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_softmax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_layernorm, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(dnnl_reorder, 1)>());
    }
};

inline void register_dnnl_opset_schema() {
    register_opset_schema<dnnl_opset_t>();
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
