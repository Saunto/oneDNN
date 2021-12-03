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
#include "closurize_impl.hpp"
#include <utility>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/any_map.hpp>
#include <util/utils.hpp>

namespace sc {
for_loop get_inner_for_loop(const for_loop_node_t *f);

closurize_impl_t::closurize_impl_t(
        const std::vector<define> &globals, ir_module_ptr modu)
    : modu_(std::move(modu)) {
    for (auto &g : globals) {
        globals_set_.insert(g->var_);
    }
}

func_c closurize_impl_t::dispatch(func_c f) {
    cur_func_ = f;
    return ir_visitor_t::dispatch(f);
}

expr_c closurize_impl_t::create_or_get_tensor_or_var(expr_c v) {
    expr_c newv;
    if (defined_set_.find(v) != defined_set_.end()) { return v; }
    if (globals_set_.find(v) != globals_set_.end()) { return v; }
    auto itr = captures_set_.find(v);
    if (itr == captures_set_.end()) {
        newv = v->remake();
        captures_set_.insert(std::make_pair(v, newv));
        captures_.emplace_back(v.remove_const());
        defined_set_.insert(newv);
    } else {
        newv = itr->second;
    }
    return newv;
}

stmt_c closurize_impl_t::visit(define_c v) {
    if (in_parallel_for && (v->var_.isa<var>() || v->var_.isa<tensor>())) {
        if (v->var_.isa<tensor>()) {
            v->var_->attr()["is_thread_buffer"] = true;
        }
        defined_set_.insert(v->var_);
    }
    return ir_visitor_t::visit(std::move(v));
}

expr_c closurize_impl_t::visit(tensor_c v) {
    if (in_parallel_for) { return create_or_get_tensor_or_var(v); }
    return v;
}

expr_c closurize_impl_t::visit(var_c v) {
    if (in_parallel_for) { return create_or_get_tensor_or_var(v); }
    return v;
}

stmt_c closurize_impl_t::visit(assign_c v) {
    auto ret = ir_visitor_t::visit(v);
    if (in_parallel_for) {
        // If the IR assigns values to the captured
        // variable, throw en error
        COMPILE_ASSERT(captures_set_.find(v->var_) == captures_set_.end(),
                "Assigning to captured vars: " << v);
    }
    return ret;
}

stmt_c closurize_impl_t::visit(for_loop_c v) {
    if (v->kind_ == for_type::PARALLEL
            || v->kind_ == for_type::GROUPED_PARALLEL) {
        COMPILE_ASSERT(!in_parallel_for,
                "Cannot have parallel for in parallel for: " << v);
        in_parallel_for = true;
        stmt_c body;
        std::vector<call_node::parallel_attr_t> attr;
        if (v->kind_ == for_type::PARALLEL) {
            dispatch(v->var_);
            attr.emplace_back(v->iter_begin_, v->iter_end_, v->step_);
            body = dispatch(v->body_);
            assert(!captures_.empty());
        } else {
            for_loop_c cur = v;
            for (int i = 0;; i++) {
                COMPILE_ASSERT(cur.defined(),
                        "Expecting 4 nested loops for GROUPED_PARALLEL: " << v);
                dispatch(cur->var_);
                // todo: add checking to throw an error if iter_begin_,
                // iter_end_ and step_ is using outer loop itervars
                attr.emplace_back(cur->iter_begin_, cur->iter_end_, cur->step_);
                if (i == 3) { break; }
                cur = get_inner_for_loop(cur.get());
            }
            body = dispatch(cur->body_);
            assert(captures_.size() >= 4);
        }
        std::vector<expr_c> params;
        for (size_t i = 0; i < captures_.size(); i++) {
            auto itr = captures_set_.find(captures_[i]);
            assert(itr != captures_set_.end());
            params.emplace_back(itr->second);
        }

        // all captured variables are in captures_
        func_t func_target = make_closure_func(
                cur_func_->name_ + "0_closure_" + std::to_string(kernel_cnt++),
                std::move(params), std::move(body), attr);

        in_parallel_for = false;
        captures_set_.clear();
        defined_set_.clear();

        auto ret = make_parallel_call(func_target, captures_, std::move(attr));
        captures_.clear();
        return ret;
    }
    if (in_parallel_for) { defined_set_.insert(v->var_); }
    return ir_visitor_t::visit(std::move(v));
}

} // namespace sc
