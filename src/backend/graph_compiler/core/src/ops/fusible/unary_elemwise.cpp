/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include <assert.h>

#include <utility>
#include "unary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <util/utils.hpp>

namespace sc {

expr relu_op_t::compute_element(expr in, int mask_count, float mask_value) {
    return builder::make_max(
            in, make_expr<constant_node>((int64_t)0, in->dtype_));
}

expr round_op_t::compute_element(expr in, int mask_count, float mask_value) {
    return builder::make_round(in);
}

expr sigmoid_op_t::compute_element(expr in, int mask_count, float mask_value) {
    auto bld = builder::get_current_builder();
    // constants
    auto lanes = in->dtype_.lanes_;
    expr f_one = make_expr<constant_node>(1.0f, sc_data_type_t::f32(lanes));
    expr sign_mask = make_expr<constant_node>(
            0x80000000UL, sc_data_type_t::u32(lanes));
    // temp vars
    auto f_neg_x = builder::make_var(
            sc_data_type_t::f32(lanes), "f_neg_x" + fusion_create_var_idx());
    bld->push_var_tensor_def(f_neg_x);

    auto f_exp_neg_x = builder::make_var(sc_data_type_t::f32(lanes),
            "f_exp_neg_x" + fusion_create_var_idx());
    bld->push_var_tensor_def(f_exp_neg_x);

    // make negative x
    bld->push_assign(f_neg_x,
            builder::make_reinterpret(
                    builder::make_int_xor(builder::make_reinterpret(in,
                                                  sc_data_type_t::u32(lanes)),
                            sign_mask),
                    sc_data_type_t::f32(lanes)));

    // out = 1 / ( 1 + exp(-x) )
    bld->push_assign(f_exp_neg_x, builder::make_exp(f_neg_x));

    return builder::make_div(f_one, f_one + f_exp_neg_x);
}

expr exp_op_t::compute_element(expr in, int mask_count, float mask_value) {
    return builder::make_exp(in, mask_count);
}

expr tanh_op_t::compute_element(expr in, int mask_count, float mask_value) {
    auto lanes = in->dtype_.lanes_;
#define DECL_VEC_CONSTANT(name, dtype, value) \
    expr name = make_expr<constant_node>(value, sc_data_type_t::dtype(lanes));

// clang-format off
// NOLINTNEXTLINE
#define DECL_VEC_VAR(name, dtype) auto name = builder::make_var( \
            sc_data_type_t::dtype(lanes), #name + fusion_create_var_idx()); \
    builder::get_current_builder()->push_var_tensor_def(name);
// clang-format on
#define DECL_CONSTANT(name, dtype, value) \
    expr name = make_expr<constant_node>(value, datatypes::dtype);
// clang-format off
// NOLINTNEXTLINE
#define DECL_VAR(name, dtype) auto name = builder::make_var( \
            datatypes::dtype, #name + fusion_create_var_idx()); \
    builder::get_current_builder()->push_var_tensor_def(name);
    // clang-format on

    auto bld = builder::get_current_builder();
    DECL_VEC_CONSTANT(uint_saturate_ubound, u32, 0x41b00000UL);
    DECL_VEC_CONSTANT(positive_mask, u32, 0x7fffffffUL);
    DECL_VEC_CONSTANT(sign_mask, u32, 0x80000000UL);
    DECL_VEC_CONSTANT(f_one, f32, 1.0f);
    DECL_VEC_CONSTANT(f_two, f32, 2.0f);
    DECL_VEC_VAR(abs_a, u32);
    DECL_VEC_VAR(sign, u32);
    DECL_VEC_VAR(f_abs_a, f32);
    DECL_VEC_VAR(f_2a, f32);
    DECL_VEC_VAR(f_exp_2a, f32);
    DECL_VEC_VAR(f_tmp, f32);
    DECL_VEC_VAR(f_out, f32);

    bld->push_assign(abs_a,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    positive_mask));
    bld->push_assign(f_abs_a,
            builder::make_reinterpret(abs_a, sc_data_type_t::f32(lanes)));
    bld->push_assign(sign,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    sign_mask));
    bld->push_assign(f_2a, builder::make_mul(f_abs_a, f_two));
    bld->push_assign(f_exp_2a, builder::make_exp(f_2a));
    bld->push_assign(
            f_tmp, builder::make_div(f_exp_2a - f_one, f_exp_2a + f_one));
    bld->push_assign(f_out,
            builder::make_select(abs_a > uint_saturate_ubound, f_one, f_tmp));
    return builder::make_reinterpret(
            builder::make_int_xor(builder::make_reinterpret(
                                          f_out, sc_data_type_t::u32(lanes)),
                    sign),
            sc_data_type_t::f32(lanes));

#undef DECL_VEC_CONSTANT
#undef DECL_VEC_VAR
#undef DECL_CONSTANT
#undef DECL_VAR
}

expr erf_op_t::compute_element(expr in, int mask_count, float mask_value) {
    auto lanes = in->dtype_.lanes_;

    auto bld = builder::get_current_builder();
    expr const_a1 = make_expr<constant_node>(
            0.254829592f, sc_data_type_t::f32(lanes));
    expr const_a2 = make_expr<constant_node>(
            -0.284496736f, sc_data_type_t::f32(lanes));
    expr const_a3 = make_expr<constant_node>(
            1.421413741f, sc_data_type_t::f32(lanes));
    expr const_a4 = make_expr<constant_node>(
            -1.453152027f, sc_data_type_t::f32(lanes));
    expr const_a5 = make_expr<constant_node>(
            1.061405429f, sc_data_type_t::f32(lanes));
    expr ONE_f = make_expr<constant_node>(1.0f, sc_data_type_t::f32(lanes));
    expr ZERO_f = make_expr<constant_node>(0.0f, sc_data_type_t::f32(lanes));
    expr const_p
            = make_expr<constant_node>(0.3275911f, sc_data_type_t::f32(lanes));
    expr sign_mask = make_expr<constant_node>(
            0x80000000UL, sc_data_type_t::u32(lanes));

    auto temp = builder::make_var(
            sc_data_type_t::f32(lanes), "temp" + fusion_create_var_idx());
    auto Q = builder::make_var(
            sc_data_type_t::f32(lanes), "Q" + fusion_create_var_idx());
    auto t = builder::make_var(
            sc_data_type_t::f32(lanes), "t" + fusion_create_var_idx());
    auto result = builder::make_var(
            sc_data_type_t::f32(lanes), "result" + fusion_create_var_idx());
    auto sign = builder::make_var(
            sc_data_type_t::u32(lanes), "sign" + fusion_create_var_idx());
    bld->push_var_tensor_def(temp);
    bld->push_var_tensor_def(Q);
    bld->push_var_tensor_def(t);
    bld->push_var_tensor_def(result);
    bld->push_var_tensor_def(sign);

    bld->push_assign(sign,
            builder::make_int_and(
                    builder::make_reinterpret(in, sc_data_type_t::u32(lanes)),
                    sign_mask));
    bld->push_assign(temp, builder::make_abs(in));
    bld->push_assign(Q, ZERO_f - builder::make_exp(ZERO_f - in * in));
    bld->push_assign(t,
            builder::make_div(
                    ONE_f, builder::make_fmadd(const_p, temp, ONE_f)));
    bld->push_assign(temp, builder::make_mul(Q, t));
    bld->push_assign(result, const_a5);
    bld->push_assign(result, builder::make_fmadd(result, t, const_a4));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a3));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a2));
    bld->push_assign(result, builder::make_fmadd(result, t, const_a1));
    bld->push_assign(result, builder::make_fmadd(result, temp, ONE_f));

    return builder::make_reinterpret(
            builder::make_int_xor(sign,
                    builder::make_reinterpret(
                            result, sc_data_type_t::u32(lanes))),
            sc_data_type_t::f32(lanes));
}

expr squared_root_op_t::compute_element(
        expr in, int mask_count, float mask_value) {
    if (reciprocal_) { return builder::make_rsqrt(in); }
    return builder::make_sqrt(in);
}

cast_op_t::cast_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : unary_elementwise_op_t("cast", ins, outs, attrs) {
    dtype_ = attrs.get<sc_data_type_t>("dtype");
    saturated_ = attrs.get_or_else("saturated", false);
    info_.outputs_[0]->details_.dtype_ = dtype_;
    info_.tensor_share_info_.clear();
}

cast_op_t::cast_op_t(
        graph_tensor_ptr v, sc_data_type_t out_dtype, bool saturated)
    : unary_elementwise_op_t(std::move(v), "cast")
    , dtype_(out_dtype)
    , saturated_(saturated) {
    info_.outputs_[0]->details_.dtype_ = out_dtype;
    info_.tensor_share_info_.clear();
}

expr cast_op_t::compute_element(expr in, int mask_count, float mask_value) {
    sc_data_type_t vectorize_out_dtype = dtype_;
    vectorize_out_dtype.lanes_ = in->dtype_.lanes_;
    return saturated_ ? builder::make_saturated_cast(in, vectorize_out_dtype)
                      : builder::make_cast(vectorize_out_dtype, in);
}

expr clamp_op_t::compute_element(expr in, int mask_count, float mask_value) {
    auto dtype = in->dtype_;
    COMPILE_ASSERT(dtype.type_code_ == sc_data_etype::F32,
            "clamp_op_t currently only supports fp32");
    float clamp_min = attrs_.get<float>("clamp_min");
    float clamp_max = attrs_.get<float>("clamp_max");
    return builder::make_max(
            builder::make_min(in, make_expr<constant_node>(clamp_max, dtype)),
            make_expr<constant_node>(clamp_min, dtype));
}

OP_REGISTER(sigmoid_op_t, sigmoid)
OP_REGISTER(exp_op_t, exp)
OP_REGISTER(erf_op_t, erf)
OP_REGISTER(tanh_op_t, tanh)
OP_REGISTER(relu_op_t, relu)
OP_REGISTER(round_op_t, round)
OP_REGISTER(squared_root_op_t, squared_root)
OP_REGISTER(cast_op_t, cast)
OP_REGISTER(clamp_op_t, clamp)

} // namespace sc
