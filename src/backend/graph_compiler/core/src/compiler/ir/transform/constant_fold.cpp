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
#include <unordered_map>

#include <utility>
#include <vector>
#include "../builder.hpp"
#include "../util_module_passes.hpp"
#include "../visitor.hpp"
#include "constant_fold.hpp"
#include <compiler/ir/ir_comparer.hpp>
#include <util/utils.hpp>

namespace sc {
namespace constant_folding {

template <typename T>
expr create_cast(sc_data_type_t to_dtype, type_category to_cate, T v) {
    switch (to_cate) {
        case CATE_FLOAT: {
            float outv = static_cast<float>(v);
            return make_expr<constant_node>(outv, to_dtype);
            break;
        }
        case CATE_INT: {
            int64_t outv = static_cast<int64_t>(v);
            return make_expr<constant_node>(outv, to_dtype);
            break;
        }
        case CATE_UINT: {
            uint64_t outv = static_cast<uint64_t>(v);
            return make_expr<constant_node>(outv, to_dtype);
            break;
        }
        default: COMPILE_ASSERT(0, "Bad cast to " << to_dtype); return expr();
    }
}

template <typename T>
union_val make_val(T) = delete;

union_val make_val(float t) {
    union_val a;
    a.f32 = t;
    return a;
}

union_val make_val(uint64_t t) {
    union_val a;
    a.u64 = t;
    return a;
}

union_val make_val(int64_t t) {
    union_val a;
    a.s64 = t;
    return a;
}

union_val make_val(bool t) {
    union_val a;
    a.u64 = t ? 1 : 0;
    return a;
}

bool is_const_equal_to(const constant_c &v, int64_t V) {
    auto cate = get_type_category(v->dtype_);
    switch (cate) {
        case CATE_FLOAT: {
            float outv = v->value_[0].f32;
            return outv == V;
        }
        case CATE_INT: {
            int64_t outv = v->value_[0].s64;
            return outv == V;
        }
        case CATE_UINT: {
            uint64_t outv = v->value_[0].u64;
            return outv == static_cast<uint64_t>(V);
        }
        default: assert(0 && "Bad category"); return false;
    }
}

bool execute_logic_binary(sc_expr_type op, bool a, bool b) {
    switch (op) {
        case sc_expr_type::logic_and: return (a && b);
        case sc_expr_type::logic_or: return (a || b);
        default: assert(0 && "Unknown logic OP"); return false;
    }
}

float execute_mod(float a, float b) {
    COMPILE_ASSERT(0, "%% cannot be applied on float type");
    return 0;
}

template <typename T>
T execute_mod(T a, T b) {
    return a % b;
}

static float execute_and(float a, float b) {
    COMPILE_ASSERT(0, "& cannot be applied on float type");
    return 0;
}

static float execute_or(float a, float b) {
    COMPILE_ASSERT(0, "| cannot be applied on float type");
    return 0;
}

template <typename T>
T execute_and(T a, T b) {
    return a & b;
}

template <typename T>
T execute_or(T a, T b) {
    return a | b;
}

template <typename T>
union_val execute_binary(sc_expr_type op, intrin_type intrin_op, T a, T b) {
    switch (op) {
        case sc_expr_type::add: return make_val(a + b);
        case sc_expr_type::sub: return make_val(a - b);
        case sc_expr_type::mul: return make_val(a * b);
        case sc_expr_type::div: return make_val(a / b);
        case sc_expr_type::mod: return make_val(execute_mod(a, b));
        case sc_expr_type::intrin_call: {
            switch (intrin_op) {
                case intrin_type::min: return make_val(a < b ? a : b);
                case intrin_type::max: return make_val(a > b ? a : b);
                case intrin_type::int_and: return make_val(execute_and(a, b));
                case intrin_type::int_or: return make_val(execute_or(a, b));
                default: assert(0 && "Unknown OP");
            }
        }
        case sc_expr_type::cmp_eq: return make_val(a == b);
        case sc_expr_type::cmp_ne: return make_val(a != b);
        case sc_expr_type::cmp_lt: return make_val(a < b);
        case sc_expr_type::cmp_le: return make_val(a <= b);
        case sc_expr_type::cmp_gt: return make_val(a > b);
        case sc_expr_type::cmp_ge: return make_val(a >= b);
        default: assert(0 && "Unknown OP"); return make_val(false);
    }
}

expr compute_constexpr(
        const constant_c &cl, const constant_c &cr, const expr_c &parent) {
    if (cl->is_vector() || cr->is_vector()) return parent.remove_const();
    COMPILE_ASSERT(cl->dtype_ == cr->dtype_,
            "LHS and RHS should have the same type: " << parent);
    if (parent.instanceof <logic>()) {
        COMPILE_ASSERT(cl->dtype_ == datatypes::boolean,
                "logic op should have boolean operands: " << parent);
        bool res = execute_logic_binary(
                parent->node_type_, cl->value_[0].u64, cr->value_[0].u64);
        return make_expr<constant_node>(
                static_cast<uint64_t>(res), datatypes::boolean);
    }
    type_category ty = get_type_category(cl->dtype_);
    auto op = parent->node_type_;
    intrin_type intrin_op = intrin_type::NUM_INTRINSICS;
    if (op == intrin_call_node::type_code_)
        intrin_op = parent.static_as<intrin_call_c>()->type_;
    union_val val;
    switch (ty) {
        case CATE_FLOAT:
            val = execute_binary(
                    op, intrin_op, cl->value_[0].f32, cr->value_[0].f32);
            break;
        case CATE_UINT:
            val = execute_binary(
                    op, intrin_op, cl->value_[0].u64, cr->value_[0].u64);
            break;
        case CATE_INT:
            val = execute_binary(
                    op, intrin_op, cl->value_[0].s64, cr->value_[0].s64);
            break;
        default:
            COMPILE_ASSERT(0, "Type of binary op: " << parent);
            return expr();
    }
    return make_expr<constant_node>(val, parent->dtype_);
}

bool is_op_commutative_and_associative(const expr_c &v) {
    if (v->node_type_ == sc_expr_type::intrin_call) {
        switch (v.static_as<intrin_call_c>()->type_) {
            case intrin_type::max:
            case intrin_type::min:
            case intrin_type::int_and:
            case intrin_type::int_or: return true;
            default: return false;
        }
    }
    switch (v->node_type_) {
        case sc_expr_type::add:
        case sc_expr_type::mul:
        case sc_expr_type::logic_and:
        case sc_expr_type::logic_or: return true;
        default: return false;
    }
}

std::pair<expr_c, expr_c> get_operand_from_binary(const expr_c &a) {
    if (a.instanceof <intrin_call_c>()) {
        auto v = a.static_as<intrin_call_c>();
        return std::make_pair(v->args_[0], v->args_[1]);
    }
    if (a.instanceof <binary_c>()) {
        auto v = a.static_as<binary_c>();
        return std::make_pair(v->l_, v->r_);
    }
    if (a.instanceof <cmp_c>()) {
        auto v = a.static_as<cmp_c>();
        return std::make_pair(v->l_, v->r_);
    }
    assert(a.instanceof <logic_c>());
    auto v = a.static_as<logic_c>();
    return std::make_pair(v->l_, v->r_);
}

bool fold_special_consts(expr_c &orig, expr_c l, const constant_c &r) {
    if (r->is_vector()) return false;
    sc_expr_type op = orig->node_type_;
    if (r->dtype_ == datatypes::boolean) {
        bool val = r->value_[0].u64;
        switch (op) {
            case sc_expr_type::logic_and:
                if (val) {
                    // x && 1 = x
                    orig = std::move(l);
                    return true;
                } else {
                    // X && 0 = 0
                    orig = make_expr<constant_node>(
                            uint64_t(0), datatypes::boolean);
                    return true;
                }
                break;
            case sc_expr_type::logic_or:
                if (val) {
                    // x || 1 = 1
                    orig = make_expr<constant_node>(
                            uint64_t(1), datatypes::boolean);
                    return true;
                } else {
                    // X || 0 = X
                    orig = std::move(l);
                    return true;
                }
                break;
            default: {
            };
        }
        return false;
    }

    if (is_const_equal_to(r, 0)) {
        switch (op) {
            case sc_expr_type::add:
            case sc_expr_type::sub: orig = std::move(l); return true;
            case sc_expr_type::mul:
                orig = make_expr<constant_node>(uint64_t(0), orig->dtype_);
                return true;
            default: {
            };
        }
    }
    if (is_const_equal_to(r, 1)) {
        switch (op) {
            case sc_expr_type::mul:
            case sc_expr_type::div: orig = std::move(l); return true;
            case sc_expr_type::mod:
                orig = make_expr<constant_node>(uint64_t(0), orig->dtype_);
                return true;
            default: {
            };
        }
    }
    return false;
}
} // namespace constant_folding

using namespace constant_folding;

/**
 * It will do the following (c as constant, "+" as an example):
 * c1 + c2 => c3
 * c + x => x + c
 * (x + c1) + c2 => x + (c1 + c2)
 * (x + c) + y => (x + y) + c
 * x + (y + c) => (x + y) + c
 * (x + c1) + (y + c2) => (x + y) + (c1 + c2)
 *
 * Also fold special expr:
 * a (+ - * && ||) 0/false
 * a (* / % && ||) 1/true
 * a (- / % && || max min > >= < <= == !=) a
 * */
class constant_fold_t : public ir_consistent_visitor_t {
public:
    using ir_consistent_visitor_t::dispatch;
    using ir_consistent_visitor_t::visit;
    // a comparer with strict var/tensor comparison
    ir_comparer cmper;
    constant_fold_t() : cmper(false, true, true, false) {}

    bool is_same_op(expr_c &v1, expr_c &v2) {
        if (v1->node_type_ != v2->node_type_) return false;
        if (v1->node_type_ == sc_expr_type::intrin_call)
            return v1.static_as<intrin_call_c>()->type_
                    == v2.static_as<intrin_call_c>()->type_;
        return true;
    }

    // try to rotate by the rotation rule.
    // returns true if rotation succeed
    bool try_rotate_const(expr_c &parent, expr_c &l, expr_c &r) {
        if (!is_op_commutative_and_associative(parent)) return false;
        if (l.isa<constant>() && !r.isa<constant>()) {
            // c + x => x + c
            std::swap(l, r);
            return true;
        }
        if (is_same_op(parent, l) && !l.isa<constant>() && r.isa<constant>()) {
            // (x + c1) + c2 => x + (c1 + c2)
            auto v = get_operand_from_binary(l);
            if (v.second.isa<constant>()) {
                auto c1 = v.second.static_as<constant_c>();
                r = compute_constexpr(c1, r.static_as<constant_c>(), parent);
                l = v.first;
                return true;
            }
        }
        if (!l.isa<constant>() && !r.isa<constant>()) {
            if (is_same_op(parent, l) && !is_same_op(parent, r)) {
                auto v = get_operand_from_binary(l);
                if (v.second.isa<constant>()) {
                    // (x + c) + y => (x + y) + c
                    l = builder::remake_binary(v.first, r, parent);
                    r = v.second;
                    return true;
                }
            }
            if (!is_same_op(parent, l) && is_same_op(parent, r)) {
                auto v = get_operand_from_binary(r);
                if (v.second.isa<constant>()) {
                    // x + (y + c) => (x + y) + c
                    l = builder::remake_binary(l, v.first, parent);
                    r = v.second;
                    return true;
                }
            }
            if (is_same_op(parent, l) && is_same_op(parent, r)) {
                auto vl = get_operand_from_binary(l);
                auto vr = get_operand_from_binary(r);
                if (vl.second.isa<constant>() && vr.second.isa<constant>()) {
                    // (x + c1) + (y + c2) => (x + y) + (c1 + c2)
                    l = builder::remake_binary(vl.first, vr.first, parent);
                    r = compute_constexpr(vl.second.checked_as<constant>(),
                            vr.second.checked_as<constant>(), parent);
                    return true;
                }
            }
        }
        return false;
    }

    // fold expr like a-a a/a a%a a&&a a||a min(a,a) max(a,a)
    // a!=a a>a ...
    bool fold_special_exprs(expr_c &parent, expr_c lhs, const expr_c &rhs) {
        switch (parent->node_type_) {
            case sc_expr_type::sub:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                break;
            case sc_expr_type::mod:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                if (rhs.isa<constant_c>()) {
                    int rv1 = get_const_as_int(rhs.checked_as<constant_c>());
                    // fold (x * nC) % C = 0
                    if (lhs->node_type_ == sc_expr_type::mul) {
                        auto rhs_of_lhs = get_operand_from_binary(lhs).second;
                        if (rhs_of_lhs.isa<constant_c>()) {
                            int rv2 = get_const_as_int(
                                    rhs_of_lhs.checked_as<constant_c>());
                            if (rv2 % rv1 == 0) {
                                parent = make_expr<constant_node>(
                                        0UL, parent->dtype_);
                                return true;
                            }
                        }
                    }
                    // fold (x %C) % C = x % C
                    else if (lhs->node_type_ == sc_expr_type::mod) {
                        auto r_l = get_operand_from_binary(lhs);
                        auto rhs_of_lhs = r_l.second;
                        if (rhs_of_lhs.isa<constant_c>()) {
                            int rv2 = get_const_as_int(
                                    rhs_of_lhs.checked_as<constant_c>());
                            if (rv2 == rv1) {
                                parent = builder::make_mod(r_l.first, rhs);
                                return true;
                            }
                        }
                    }
                }
                break;
            case sc_expr_type::div:
                if (cmper.compare(lhs, rhs)) {
                    switch (get_type_category(parent->dtype_)) {
                        case CATE_INT:
                        case CATE_UINT:
                            parent = make_expr<constant_node>(
                                    1UL, parent->dtype_);
                            return true;
                        case CATE_FLOAT:
                            parent = make_expr<constant_node>(
                                    1.0f, parent->dtype_);
                            return true;
                        default: assert(0 && "Bad type"); return false;
                    }
                }
                break;
            case sc_expr_type::intrin_call:
                // todo(xxx): fold &0 |1
                bool can_fold;
                switch (parent.static_as<intrin_call>()->type_) {
                    case intrin_type::max:
                    case intrin_type::min:
                    case intrin_type::int_and:
                    case intrin_type::int_or: can_fold = true; break;
                    default: can_fold = false;
                }
                if (!can_fold) break;
                // if can_fold, fall through
            case sc_expr_type::logic_and:
            case sc_expr_type::logic_or:
                if (cmper.compare(lhs, rhs)) {
                    parent = std::move(lhs);
                    return true;
                }
                break;
            case sc_expr_type::cmp_eq:
            case sc_expr_type::cmp_le:
            case sc_expr_type::cmp_ge:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(1UL, parent->dtype_);
                    return true;
                }
                break;
            case sc_expr_type::cmp_ne:
            case sc_expr_type::cmp_lt:
            case sc_expr_type::cmp_gt:
                if (cmper.compare(lhs, rhs)) {
                    parent = make_expr<constant_node>(0UL, parent->dtype_);
                    return true;
                }
                break;
            default: break;
        }
        return false;
    }

    /** expand Polynomial function
     *  e.g. ((a+b)*c+d)*e = a*c*e+b*c*e+d*e
     *                  *
     *                 / \
     *                +   e
     *               / \
     *              *   d
     *             / \
     *            +   c
     *           / \
     *          a   b
     * */
    expr_c expand_polynomial(expr_c parent) {
        switch (parent->node_type_) {
            case sc_expr_type::mul:
            case sc_expr_type::div:
            case sc_expr_type::mod: {
                // TODO(xxx): support (a+b)*(c+d)
                auto l_r = get_operand_from_binary(parent);
                if (!l_r.second.isa<constant_c>()) {
                    break;
                } else {
                    constant_c rv = l_r.second.checked_as<constant_c>();
                    switch (l_r.first->node_type_) {
                        case sc_expr_type::add:
                        case sc_expr_type::sub: {
                            // TODO(xxx): special case for distribution law of
                            // Integer division
                            if (parent->node_type_ == sc_expr_type::div) {
                                auto l_r = get_operand_from_binary(parent);
                                return builder::remake_binary(
                                        expand_polynomial(l_r.first),
                                        expand_polynomial(l_r.second), parent);
                            } else {
                                auto next_lr
                                        = get_operand_from_binary(l_r.first);
                                auto new_parent = builder::remake_binary(
                                        expand_polynomial(
                                                builder::remake_binary(
                                                        next_lr.first, rv,
                                                        parent)),
                                        expand_polynomial(
                                                builder::remake_binary(
                                                        next_lr.second, rv,
                                                        parent)),
                                        l_r.first);
                                if (parent->node_type_ == sc_expr_type::mod)
                                    return builder::remake_binary(
                                            new_parent, l_r.second, parent);
                                else {
                                    return new_parent;
                                }
                            }
                        }
                        case sc_expr_type::mul:
                        case sc_expr_type::div:
                        case sc_expr_type::mod: {
                            if (parent->node_type_ == sc_expr_type::mod) {
                                if (fold_special_exprs(
                                            parent, l_r.first, l_r.second)) {
                                    return expand_polynomial(parent);
                                }
                            }
                            auto next_lr = get_operand_from_binary(l_r.first);
                            // folding
                            if (next_lr.second.isa<constant_c>()) {
                                auto new_parent = expand_polynomial(
                                        builder::remake_binary(
                                                expand_polynomial(
                                                        next_lr.first),
                                                next_lr.second, l_r.first));
                                return builder::remake_binary(
                                        new_parent, l_r.second, parent);
                            } else {
                                break;
                            }
                        }
                        default: return parent;
                    }
                }
                break;
            }
            case sc_expr_type::add:
            case sc_expr_type::sub: {
                auto l_r = get_operand_from_binary(parent);
                return builder::remake_binary(expand_polynomial(l_r.first),
                        expand_polynomial(l_r.second), parent);
            }
            default: return parent;
        }
        return parent;
    }

    expr_c fold_binary_impl(
            expr_c parent, const expr_c &lhs, const expr_c &rhs) {
        auto l = dispatch(lhs);
        auto r = dispatch(rhs);
        bool is_vector
                = (l.isa<constant>() && l.static_as<constant_c>()->is_vector())
                || (r.isa<constant>()
                        && r.static_as<constant_c>()->is_vector());
        if (!is_vector) {
            if (l.isa<constant>() && r.isa<constant>()) {
                auto cl = l.static_as<constant_c>();
                auto cr = r.static_as<constant_c>();
                return compute_constexpr(cl, cr, parent);
            }
            try_rotate_const(parent, l, r);
            if (r.isa<constant>()) {
                if (fold_special_consts(parent, l, r.static_as<constant>())) {
                    return parent;
                }
            }
            if (fold_special_exprs(parent, l, r)) { return parent; }
        }
        if (!l.ptr_same(lhs) || !r.ptr_same(rhs)) {
            return builder::remake_binary(l, r, parent);
        }
        return parent;
    }

    // run fold_binary_impl repeatedly on the expr until no changes happen
    expr_c fold_binary(expr_c parent) {
        expr_c old = parent;
        auto parent_type = parent->node_type_;
        constexpr int max_iter = 5000;
        int loop_cnt = 0;
        for (;;) {
            auto l_r = get_operand_from_binary(parent);
            expr_c ret = fold_binary_impl(parent, l_r.first, l_r.second);
            bool isT = ret->node_type_ == parent_type;
            if (ret.ptr_same(old) || !isT) { return ret; }
            parent = ret;
            old = std::move(ret);
            loop_cnt++;
            COMPILE_ASSERT(loop_cnt < max_iter,
                    "Constant folder reaches max iteration time. Either the "
                    "expression is too complicated or it is a bug of the "
                    "constant folder.")
        }
    }

    expr_c visit(cast_c v) override {
        auto in = dispatch(v->in_);
        bool changed = !in.ptr_same(v->in_);
        if (in.isa<constant>()) {
            auto inconst = in.as<constant_c>();
            if (inconst->is_vector()) return expr();
            type_category fromty = get_type_category_nothrow(inconst->dtype_);
            type_category toty = get_type_category_nothrow(v->dtype_);
            if (fromty != CATE_OTHER && toty != CATE_OTHER) {
                switch (fromty) {
                    case CATE_FLOAT:
                        return create_cast(
                                v->dtype_, toty, inconst->value_[0].f32);
                        break;
                    case CATE_UINT:
                        return create_cast(
                                v->dtype_, toty, inconst->value_[0].u64);
                        break;
                    case CATE_INT:
                        return create_cast(
                                v->dtype_, toty, inconst->value_[0].s64);
                        break;
                    default:
                        COMPILE_ASSERT(0, "Bad cast from " << inconst->dtype_);
                        return expr();
                }
            }
        }
        if (changed) {
            return copy_attr(*v, builder::make_cast(v->dtype_, in));
        } else {
            return v;
        }
    }

    expr_c visit(binary_c v) override { return fold_binary(v); }
    expr_c visit(cmp_c v) override { return fold_binary(v); }
    expr_c visit(logic_c v) override { return fold_binary(v); }
    expr_c visit(intrin_call_c v) override {
        auto ret = ir_consistent_visitor_t::visit(std::move(v));
        if (ret.isa<intrin_call>()) {
            auto node = ret.static_as<intrin_call_c>();
            switch (node->type_) {
                case intrin_type::max:
                case intrin_type::min:
                case intrin_type::int_and:
                case intrin_type::int_or: return fold_binary(node);
                default: break;
            }
        }
        return ret;
    }
    expr_c visit(logic_not_c v) override {
        auto in = dispatch(v->in_);
        bool changed = !in.ptr_same(v->in_);
        if (in.isa<constant>()) {
            auto inconst = in.as<constant>();
            if (inconst->is_vector()) return v;
            COMPILE_ASSERT(inconst->dtype_ == datatypes::boolean,
                    "logic_not should have a boolean operand: " << v);
            uint64_t v = inconst->value_[0].u64 ? 0 : 1;
            return make_expr<constant_node>(v, datatypes::boolean);
        }
        if (changed) {
            return copy_attr(*v, builder::make_logic_not(in));
        } else {
            return v;
        }
    }

    stmt_c visit(if_else_c v) override {
        auto cond = dispatch(v->condition_);
        auto thencase = dispatch(v->then_case_);

        stmt_c elsecase;
        if (v->else_case_.defined()) elsecase = dispatch(v->else_case_);
        bool changed = !cond.ptr_same(v->condition_)
                || !elsecase.ptr_same(v->else_case_)
                || !thencase.ptr_same(v->then_case_);
        if (v->condition_.isa<constant>()) {
            assert(!v->condition_.as<constant>()->is_vector());
            COMPILE_ASSERT(v->condition_->dtype_ == datatypes::boolean,
                    "IfElse node expects an boolean expr as the condition, got "
                            << v->condition_->dtype_ << " expr = " << v);
            bool val = v->condition_.as<constant>()->value_[0].u64;
            if (val) {
                return v->then_case_;
            } else {
                return make_stmt<stmts_node_t>(std::vector<stmt>());
            }
        }
        if (changed) {
            return copy_attr(*v,
                    builder::make_if_else_unattached(cond, thencase, elsecase));
        }
        return v;
    }
};

func_c constant_folder_t::operator()(func_c f) {
    constant_fold_t pass;
    return pass.dispatch(std::move(f));
}

stmt_c constant_folder_t::operator()(stmt_c f) {
    constant_fold_t pass;
    return pass.dispatch(std::move(f));
}

expr_c constant_folder_t::operator()(expr_c f) {
    constant_fold_t pass;
    return pass.dispatch(std::move(f));
}

/**
 *  this feature is currently used to fold the index of reshape/reorder output,
 * so additional folding is added before and after expand_polynomial. TODO: move
 *  this feature to constant folding pass
 *  @param f: original polynomial expr.
 *  @param max_iter: maximum iteration time, default is one.
 * */
expr_c constant_folder_t::expand_polynomial(expr_c f, int max_iter) {
    constant_fold_t pass;
    auto ret = pass.dispatch(std::move(f));
    for (int i = 0; i < max_iter; i++) {
        auto old = ret;
        ret = pass.expand_polynomial(old);
        if (ret.ptr_same(old)) { break; }
    }
    return pass.dispatch(ret);
}

const_ir_module_ptr constant_folder_t::operator()(const_ir_module_ptr f) {
    constant_fold_t pass;
    return dispatch_module_on_visitor(&pass, f);
}

} // namespace sc
