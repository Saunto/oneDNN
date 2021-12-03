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
#include "ir_copy.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../viewer.hpp"
#include "ir_copy_internal.hpp"
#include <util/any_map.hpp>

namespace sc {

expr ir_copier_impl_t::copy(const expr_c &v) {
    dispatch(v);
    return copy_attr(*v, std::move(returned_expr_));
}

stmt ir_copier_impl_t::copy(const stmt_c &v) {
    dispatch(v);
    return copy_attr(*v, std::move(returned_stmt_));
}

bool ir_copier_impl_t::find_and_return(const expr_c &v) {
    auto itr = replace_map_.find(v);
    if (itr != replace_map_.end()) {
        returned_expr_ = itr->second;
        return true;
    } else {
        // if the var/tensor is not found in the replace map
        if (!create_var_tensor_) {
            // if we use the old var/tensor
            returned_expr_ = v.remove_const();
            return true;
        }
        return false;
    }
}

ir_copier_impl_t::ir_copier_impl_t(
        std::unordered_map<expr_c, expr> &replace_map, bool create_var_tensor)
    : replace_map_(replace_map), create_var_tensor_(create_var_tensor) {}

void ir_copier_impl_t::view(constant_c v) {
    returned_expr_ = make_expr<constant_node>(v->value_, v->dtype_);
}

void ir_copier_impl_t::view(var_c v) {
    if (find_and_return(v)) return;
    returned_expr_ = builder::make_var(v->dtype_, v->name_);
    replace_map_.insert(std::make_pair(v, returned_expr_));
}

void ir_copier_impl_t::view(cast_c v) {
    returned_expr_ = builder::make_cast(v->dtype_, copy(v->in_));
}

#define COPY_BINARY(CLASS) \
    void ir_copier_impl_t::view(CLASS##_c v) { \
        returned_expr_ = builder::make_##CLASS(copy(v->l_), copy(v->r_)); \
    }

void ir_copier_impl_t::view(add_c v) {
    auto ret = builder::make_add(copy(v->l_), copy(v->r_));
    if (v->dtype_ != datatypes::undef) ret->dtype_ = v->dtype_;
    returned_expr_ = std::move(ret);
}

COPY_BINARY(sub)
COPY_BINARY(mul)
COPY_BINARY(div)
COPY_BINARY(mod)
COPY_BINARY(cmp_eq)
COPY_BINARY(cmp_lt)
COPY_BINARY(cmp_le)
COPY_BINARY(cmp_gt)
COPY_BINARY(cmp_ge)
COPY_BINARY(cmp_ne)
COPY_BINARY(logic_and)
COPY_BINARY(logic_or)

void ir_copier_impl_t::view(logic_not_c v) {
    returned_expr_ = builder::make_logic_not(copy(v->in_));
}

void ir_copier_impl_t::view(select_c v) {
    returned_expr_
            = builder::make_select(copy(v->cond_), copy(v->l_), copy(v->r_));
}

void ir_copier_impl_t::view(indexing_c v) {
    expr ptr = copy(v->ptr_);
    std::vector<expr> idx;
    idx.reserve(v->idx_.size());
    for (auto &i : v->idx_) {
        idx.emplace_back(copy(i));
    }
    expr mask;
    if (v->mask_.defined()) { mask = copy(v->mask_); }
    returned_expr_ = builder::make_indexing(ptr, idx, v->dtype_.lanes_, mask);
}

void ir_copier_impl_t::view(tensorptr_c v) {
    returned_expr_ = make_expr<tensorptr_node>(
            copy(v->base_).checked_as<indexing>(), v->shape_, v->is_slice_);
}

void ir_copier_impl_t::view(call_c v) {
    // do not copy the function AST
    std::vector<expr> args;
    args.reserve(v->args_.size());
    for (auto &i : v->args_) {
        args.emplace_back(copy(i));
    }
    returned_expr_ = make_expr<call_node>(v->func_, args,
            std::vector<call_node::parallel_attr_t> {v->para_attr_});
}

void ir_copier_impl_t::view(intrin_call_c v) {
    std::vector<expr> args;
    args.reserve(v->args_.size());
    for (auto &i : v->args_) {
        args.emplace_back(copy(i));
    }
    returned_expr_ = builder::remake_intrin_call(v, args);
    returned_expr_->dtype_ = v->dtype_;
}

void ir_copier_impl_t::view(func_addr_c v) {
    returned_expr_ = builder::make_func_addr(v->func_);
}

void ir_copier_impl_t::view(tensor_c v) {
    if (find_and_return(v)) return;
    std::vector<expr> args;
    args.reserve(v->dims_.size());
    for (auto &i : v->dims_) {
        args.emplace_back(copy(i));
    }
    returned_expr_ = builder::make_tensor(
            v->name_, args, v->elem_dtype_, v->address_space_, v->init_value_);
    replace_map_.insert(std::make_pair(v, returned_expr_));
}

void ir_copier_impl_t::view(assign_c v) {
    returned_stmt_ = make_stmt<assign_node_t>(copy(v->var_), copy(v->value_));
}

void ir_copier_impl_t::view(stmts_c v) {
    std::vector<stmt> seq;
    seq.reserve(v->seq_.size());
    for (auto &i : v->seq_) {
        seq.emplace_back(copy(i));
    }
    returned_stmt_ = make_stmt<stmts_node_t>(std::move(seq));
}

void ir_copier_impl_t::view(if_else_c v) {
    returned_stmt_ = make_stmt<if_else_node_t>(copy(v->condition_),
            copy(v->then_case_),
            v->else_case_.defined() ? copy(v->else_case_) : stmt());
}

void ir_copier_impl_t::view(evaluate_c v) {
    returned_stmt_ = make_stmt<evaluate_node_t>(copy(v->value_));
}

void ir_copier_impl_t::view(returns_c v) {
    returned_stmt_ = make_stmt<returns_node_t>(
            v->value_.defined() ? copy(v->value_) : expr());
}

void ir_copier_impl_t::view(define_c v) {
    returned_stmt_ = make_stmt<define_node_t>(copy(v->var_), v->linkage_,
            v->init_.defined() ? copy(v->init_) : expr());
}

void ir_copier_impl_t::view(for_loop_c v) {
    returned_stmt_ = make_stmt<for_loop_node_t>(copy(v->var_),
            copy(v->iter_begin_), copy(v->iter_end_), copy(v->step_),
            copy(v->body_), v->incremental_, v->kind_);
}

func_c ir_copier_impl_t::dispatch(func_c v) {
    std::vector<expr> params;
    params.reserve(v->params_.size());
    for (auto &i : v->params_) {
        params.emplace_back(copy(i));
    }
    auto body = v->body_.defined() ? copy(v->body_) : stmt();
    returned_func_ = builder::make_func(v->name_, params, body, v->ret_type_);
    if (v->attr_) {
        returned_func_->attr_ = utils::make_unique<any_map_t>(*v->attr_);
    }
    return v;
}

func_t ir_copier_impl_t::copy(const func_c &f) {
    dispatch(func_t(std::const_pointer_cast<func_base>(f)));
    return copy_attr(*f, std::move(returned_func_));
}

stmt_c ir_copier_t::operator()(const stmt_c &s) {
    ir_copier_impl_t vis(replace_map_, create_var_tensor_);
    return vis.copy(s);
}

expr_c ir_copier_t::operator()(const expr_c &s) {
    ir_copier_impl_t vis(replace_map_, create_var_tensor_);
    return vis.copy(s);
}

func_c ir_copier_t::operator()(func_c s) {
    ir_copier_impl_t vis(replace_map_, create_var_tensor_);
    return vis.copy(s);
}

} // namespace sc
