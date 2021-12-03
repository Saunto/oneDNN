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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BF16_LEGALIZE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BF16_LEGALIZE_HPP

#include <tuple>
#include <utility>
#include "../function_pass.hpp"
#include "../sc_function.hpp"
#include "../visitor.hpp"
#include <compiler/config/context.hpp>
#include <unordered_map>

namespace sc {

class bf16_promote_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    std::tuple<expr_c, expr_c> docast(
            const expr &orig_a, const expr &orig_b, bool *is_bfloat16);
    expr_c visit(binary_c v) final;
    expr_c visit(cmp_c v) final;
    expr_c visit(intrin_call_c v) final;
};

class bf16_cast_elimination_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    // need to convert bf16 var to f32
    std::unordered_map<expr_c, expr_c> cvt_map_;
    expr_c visit(cast_c v) final;
    stmt_c visit(define_c v) final;
    stmt_c visit(assign_c v) final;
    stmt_c visit(returns_c v) final;
};

/**
 * bfloat16 legalize pass.
 *
 * It will do the following (a, b as bfloat16 input, c as bfloat16 output, "+"
 * as example):
 * c = a + b => c = bf16(float(a)+float(b))
 * c = a + neg(b) => c = bf16(float(a), neg(float(b)))
 * */
class bf16_legalize_t : public function_pass_t {
public:
    bf16_legalize_t(context_ptr ctx = get_default_context())
        : ctx_(std::move(ctx)) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    expr_c operator()(expr_c f);

private:
    context_ptr ctx_;
};

} // namespace sc

#endif
