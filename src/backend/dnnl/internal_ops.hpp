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

// We define those internal used operators in this file. For those operators
// defined on API can be found at src/interface/c_types_map.hpp.

#ifndef BACKEND_DNNL_INTERNAL_OPS_HPP
#define BACKEND_DNNL_INTERNAL_OPS_HPP

#include <string>
#include <vector>

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace op_kind {

// X(s, v):
// s will be the internal op kind value, can be accessed via impl::op_kind::s.
// v will be used to define the name string of each op kind.
#define INTERNAL_OPS \
    X(bn_relu, BatchNorm_relu) \
    X(bn_bwd_relu_bwd, BatchNormBwd_reluBwd) \
    X(bn_fwd_train_relu, BatchNormFwdTrain_relu) \
    X(conv_add, Conv_add) \
    X(conv_add_elu, Conv_add_elu) \
    X(conv_add_relu, Conv_add_relu) \
    X(conv_add_relu6, Conv_add_relu6) \
    X(conv_bias, Conv_bias) \
    X(conv_bias_abs, Conv_bias_abs) \
    X(conv_bias_add, Conv_bias_add) \
    X(conv_bias_add_elu, Conv_bias_add_elu) \
    X(conv_bias_add_relu, Conv_bias_add_relu) \
    X(conv_bias_add_relu6, Conv_bias_add_relu6) \
    X(conv_bias_bn, Conv_bias_bn) \
    X(conv_bias_bn_add, Conv_bias_bn_add) \
    X(conv_bias_bn_add_relu, Conv_bias_bn_add_relu) \
    X(conv_bias_bn_relu, Conv_bias_bn_relu) \
    X(conv_bias_elu, Conv_bias_elu) \
    X(conv_bias_hardtanh, Conv_bias_hardtanh) \
    X(conv_bias_relu, Conv_bias_relu) \
    X(conv_bias_relu6, Conv_bias_relu6) \
    X(conv_bias_sigmoid, Conv_bias_sigmoid) \
    X(conv_bias_sqrt, Conv_bias_sqrt) \
    X(conv_bias_square, Conv_bias_square) \
    X(conv_bias_swish, Conv_bias_swish) \
    X(conv_bias_tanh, Conv_bias_tanh) \
    X(conv_bn, Conv_bn) \
    X(conv_bn_add, Conv_bn_add) \
    X(conv_bn_add_relu, Conv_bn_add_relu) \
    X(conv_bn_relu, Conv_bn_relu) \
    X(conv_relu, Conv_relu) \
    X(conv_depthwise, Conv_depthwise) \
    X(conv_bwd_f_biasadd_bwd, ConvBwdF_biasAddBwd) \
    X(convtranspose_fusion, ConvTranspose_fusion) \
    X(matmul_bias, MatMul_bias) \
    X(matmul_bias_add, MatMul_bias_add) \
    X(matmul_bias_add_relu, MatMul_bias_add_relu) \
    X(matmul_bias_bn, MatMul_bias_bn) \
    X(matmul_bias_elu, MatMul_bias_elu) \
    X(matmul_bias_hardtanh, MatMul_bias_hardtanh) \
    X(matmul_bias_relu, MatMul_bias_relu) \
    X(matmul_bias_relu6, MatMul_bias_relu6) \
    X(matmul_bias_gelu, MatMul_bias_gelu) \
    X(matmul_bias_sigmoid, MatMul_bias_sigmoid) \
    X(matmul_bias_swish, MatMul_bias_swish) \
    X(matmul_relu, MatMul_relu) \
    X(matmul_elu, MatMul_elu) \
    X(matmul_sigmoid, MatMul_sigmoid) \
    X(matmul_hardtanh, MatMul_hardtanh) \
    X(matmul_gelu, MatMul_gelu) \
    X(matmul_add, MatMul_add) \
    X(matmul_add_gelu, MatMul_add_gelu) \
    X(matmul_add_relu, MatMul_add_relu) \
    X(matmul_add_sigmoid, MatMul_add_sigmoid) \
    X(avgpool_add, AvgPool_add) \
    X(maxpool_add, MaxPool_add) \
    X(int8_conv, INT8_Conv) \
    X(int8_quant_wei_conv, INT8_Quant_wei_Conv) \
    X(int8_conv_bias, INT8_Conv_bias) \
    X(int8_quant_wei_conv_bias, INT8_Quant_wei_Conv_bias) \
    X(int8_conv_relu, INT8_Conv_relu) \
    X(int8_quant_wei_conv_relu, INT8_Quant_wei_Conv_relu) \
    X(int8_conv_bias_add, INT8_Conv_bias_add) \
    X(int8_conv_bias_relu, INT8_Conv_bias_relu) \
    X(int8_quant_wei_conv_bias_relu, INT8_Quant_wei_Conv_bias_relu) \
    X(int8_conv_add_relu, INT8_Conv_add_relu) \
    X(int8_quant_wei_conv_add_relu, INT8_Quant_wei_Conv_add_relu) \
    X(int8_conv_bias_add_relu, INT8_Conv_bias_add_relu) \
    X(int8_quant_wei_conv_bias_add_relu, INT8_Quant_wei_Conv_bias_add_relu) \
    X(int8_convtranspose, INT8_ConvTranspose) \
    X(int8_convtranspose_bias, INT8_ConvTranspose_bias) \
    X(int8_matmul, INT8_MatMul) \
    X(int8_quant_wei_matmul, INT8_Quant_wei_MatMul) \
    X(int8_matmul_bias, INT8_MatMul_bias) \
    X(int8_quant_wei_matmul_bias, INT8_Quant_wei_MatMul_bias) \
    X(int8_matmul_relu, INT8_MatMul_relu) \
    X(int8_quant_wei_matmul_relu, INT8_Quant_wei_MatMul_relu) \
    X(int8_matmul_bias_relu, INT8_MatMul_bias_relu) \
    X(int8_quant_wei_matmul_bias_relu, INT8_Quant_wei_MatMul_bias_relu) \
    X(int8_matmul_sigmoid, INT8_MatMul_sigmoid) \
    X(int8_quant_wei_matmul_sigmoid, INT8_Quant_wei_MatMul_sigmoid) \
    X(int8_matmul_bias_sigmoid, INT8_MatMul_bias_sigmoid) \
    X(int8_quant_wei_matmul_bias_sigmoid, INT8_Quant_wei_MatMul_bias_sigmoid) \
    X(int8_matmul_gelu, INT8_MatMul_gelu) \
    X(int8_quant_wei_matmul_gelu, INT8_Quant_wei_MatMul_gelu) \
    X(int8_matmul_bias_gelu, INT8_MatMul_bias_gelu) \
    X(int8_quant_wei_matmul_bias_gelu, INT8_Quant_wei_MatMul_bias_gelu) \
    X(int8_matmul_add, INT8_MatMul_add) \
    X(int8_quant_wei_matmul_add, INT8_Quant_wei_MatMul_add) \
    X(int8_matmul_bias_add, INT8_MatMul_bias_add) \
    X(int8_quant_wei_matmul_bias_add, INT8_Quant_wei_MatMul_bias_add) \
    X(x8s8float_matmul_add, X8S8FLOAT_MatMul_add) \
    X(x8s8float_matmul_bias_add, X8S8FLOAT_MatMul_bias_add) \
    X(mul_scales, Mul_scales) \
    X(add_zps, Add_zps) \
    X(permute, Permute) \
    X(to_group, To_group) \
    X(expand, Expand) \
    X(squeeze, Squeeze) \
    X(dnnl_convolution, Dnnl_convolution) \
    X(dnnl_convtranspose, Dnnl_convtranspose) \
    X(int8_maxpool, INT8_MaxPool) \
    X(relu_add, Relu_add) \
    X(add_relu, Add_relu) \
    X(add_sigmoid, Add_sigmoid) \
    X(add_multiply, Add_multiply) \
    X(multiply_relu, Multiply_relu) \
    X(multiply_sigmoid, Multiply_sigmoid) \
    X(maximum_relu, Maximum_relu) \
    X(maximum_sigmoid, Maximum_sigmoid) \
    X(minimum_relu, Minimum_relu) \
    X(minimum_sigmoid, Minimum_sigmoid) \
    X(x8x8float_matmul, X8X8FLOAT_MatMul) \
    X(x8s8float_matmul_bias, X8S8FLOAT_MatMul_bias) \
    X(x8s8f32_matmul_relu, X8S8F32_MatMul_relu) \
    X(x8s8f32_matmul_bias_relu, X8S8F32_MatMul_bias_relu) \
    X(x8s8f32_matmul_sigmoid, X8S8F32_MatMul_sigmoid) \
    X(x8s8f32_matmul_bias_sigmoid, X8S8F32_MatMul_bias_sigmoid) \
    X(x8s8f32_matmul_gelu, X8S8F32_MatMul_gelu) \
    X(x8s8f32_matmul_bias_gelu, X8S8F32_MatMul_bias_gelu) \
    X(multiply_add, Multiply_add) \
    X(maximum_add, Maximum_add) \
    X(minimum_add, Minimum_add) \
    X(x8s8f32_conv, X8S8F32_Conv) \
    X(x8s8f32_conv_bias, X8S8F32_Conv_bias) \
    X(x8s8f32_conv_relu, X8S8F32_Conv_relu) \
    X(x8s8f32_conv_bias_relu, X8S8F32_Conv_bias_relu) \
    X(x8s8f32_conv_add_relu, X8S8F32_Conv_add_relu) \
    X(x8s8f32_conv_bias_add_relu, X8S8F32_Conv_bias_add_relu) \
    X(x8s8f32_quant_wei_matmul_add, X8S8F32_Quant_wei_MatMul_add) \
    X(x8s8f32_quant_wei_matmul_bias_add, X8S8F32_Quant_wei_MatMul_bias_add) \
    X(x8s8f32_quant_wei_matmul, X8S8F32_Quant_wei_MatMul) \
    X(x8s8f32_quant_wei_matmul_bias, X8S8F32_Quant_wei_MatMul_bias) \
    X(x8s8f32_quant_wei_matmul_relu, X8S8F32_Quant_wei_MatMul_relu) \
    X(x8s8f32_quant_wei_matmul_bias_relu, X8S8F32_Quant_wei_MatMul_bias_relu) \
    X(x8s8f32_quant_wei_matmul_sigmoid, X8S8F32_Quant_wei_MatMul_sigmoid) \
    X(x8s8f32_quant_wei_matmul_bias_sigmoid, \
            X8S8F32_Quant_wei_MatMul_bias_sigmoid) \
    X(x8s8f32_quant_wei_matmul_gelu, X8S8F32_Quant_wei_MatMul_gelu) \
    X(x8s8f32_quant_wei_matmul_bias_gelu, X8S8F32_Quant_wei_MatMul_bias_gelu) \
    X(x8s8f32_quant_wei_conv, X8S8F32_Quant_wei_Conv) \
    X(x8s8f32_quant_wei_conv_bias, X8S8F32_Quant_wei_Conv_bias) \
    X(x8s8f32_quant_wei_conv_relu, X8S8F32_Quant_wei_Conv_relu) \
    X(x8s8f32_quant_wei_conv_bias_relu, X8S8F32_Quant_wei_Conv_bias_relu) \
    X(x8s8f32_quant_wei_conv_add_relu, X8S8F32_Quant_wei_Conv_add_relu) \
    X(x8s8f32_quant_wei_conv_bias_add_relu, \
            X8S8F32_Quant_wei_Conv_bias_add_relu) \
    X(int8_avgpool, INT8_AvgPool) \
    X(dnnl_pool, Dnnl_pool) \
    X(dnnl_u8_to_s8, Dnnl_u8_to_s8) \
    X(dnnl_bn_folding, Dnnl_bn_folding) \
    X(dnnl_conv_bwd_data, Dnnl_conv_bwd_data) \
    X(dnnl_swish, Dnnl_swish) \
    X(x8x8float_matmul_div, X8X8FLOAT_MatMul_div) \
    X(x8x8float_matmul_div_add, X8X8FLOAT_MatMul_div_add) \
    X(dnnl_batchnorm, Dnnl_batchnorm) \
    X(dnnl_binary, Dnnl_binary) \
    X(dnnl_eltwise, Dnnl_eltwise) \
    X(dnnl_shuffle, Dnnl_shuffle) \
    X(dnnl_sum, Dnnl_sum) \
    X(conv_simple_resblock, Conv_simple_resblock) \
    X(int8_MHA, INT8_MHA) \
    X(f32_MHA, F32_MHA) \
    X(chained_relu, Chained_relu) \
    X(dnnl_prelu, Dnnl_prelu)

enum {
    kDNNL_INTERNAL_OP_STARTER = 0x1234,
#define X(s, v) k##v,
    INTERNAL_OPS
#undef X
};

#define X(s, v) const op_kind_t s = static_cast<op_kind_t>(k##v);
INTERNAL_OPS
#undef X

#define X(s, v) #v,
const std::vector<std::string> internal_op_strings = {INTERNAL_OPS};
#undef X

#undef INTERNAL_OPS

} // namespace op_kind
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
