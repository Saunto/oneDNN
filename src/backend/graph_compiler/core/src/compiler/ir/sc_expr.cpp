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
#include <iostream>
#include <limits>
#include <string.h>
#include <utility>

#include "builder.hpp"
#include "intrinsics.hpp"
#include "ir_comparer.hpp"
#include "sc_expr.hpp"
#include "sc_function.hpp"
#include "visitable.hpp"
#include <common/dimensions.hpp>
#include <util/any_map.hpp>
namespace sc {

any_map_t &expr_base::attr() {
    if (!attr_) { attr_ = utils::make_unique<any_map_t>(); }
    return *attr_;
}

ostream &operator<<(ostream &os, sc_expr_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case sc::sc_expr_type::X: os << "sc_expr_type::" #X; break;

        HANDLE_CASE(undef)
        HANDLE_CASE(constant)
        HANDLE_CASE(var)
        HANDLE_CASE(cast)
        HANDLE_CASE(add)
        HANDLE_CASE(sub)
        HANDLE_CASE(mul)
        HANDLE_CASE(div)
        HANDLE_CASE(mod)
        HANDLE_CASE(cmp_eq)
        HANDLE_CASE(cmp_ne)
        HANDLE_CASE(cmp_lt)
        HANDLE_CASE(cmp_le)
        HANDLE_CASE(cmp_gt)
        HANDLE_CASE(cmp_ge)
        HANDLE_CASE(logic_and)
        HANDLE_CASE(logic_or)
        HANDLE_CASE(logic_not)
        HANDLE_CASE(select)
        HANDLE_CASE(indexing)
        HANDLE_CASE(call)
        HANDLE_CASE(tensor)
        HANDLE_CASE(tensorptr)
        HANDLE_CASE(intrin_call)
        HANDLE_CASE(func_addr)
#undef HANDLE_CASE
        default: os << "(unrecognized sc_expr_type value)"; break;
    }
    return os;
}

ostream &operator<<(ostream &os, intrin_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case sc::intrin_type::X: os << "intrin_type::" #X; break;

        HANDLE_CASE(min)
        HANDLE_CASE(max)
        HANDLE_CASE(abs)
        HANDLE_CASE(round)
        HANDLE_CASE(floor)
        HANDLE_CASE(ceil)
        HANDLE_CASE(exp)
        HANDLE_CASE(sqrt)
        HANDLE_CASE(rsqrt)
        HANDLE_CASE(reduce_add)
        HANDLE_CASE(reduce_mul)
        HANDLE_CASE(fmadd)
        HANDLE_CASE(unpack_low)
        HANDLE_CASE(unpack_high)
        HANDLE_CASE(shuffle)
        HANDLE_CASE(permute)
        HANDLE_CASE(int_and)
        HANDLE_CASE(int_or)
        HANDLE_CASE(int_xor)
        HANDLE_CASE(reinterpret)
        HANDLE_CASE(broadcast)
        HANDLE_CASE(isnan)
        HANDLE_CASE(shl)
        HANDLE_CASE(shr)
        HANDLE_CASE(brgemm)
        HANDLE_CASE(list_brgemm)
        HANDLE_CASE(NUM_INTRINSICS)
#undef HANDLE_CASE
        default: os << "(unrecognized intrin_type value)"; break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, linkage val) {
    switch (val) {
        case sc::linkage::public_global: os << "linkage::public_global"; break;
        case sc::linkage::private_global:
            os << "linkage::private_global";
            break;
        case sc::linkage::static_local: os << "linkage::static_local"; break;
        case sc::linkage::local: os << "linkage::local"; break;
    }
    return os;
}

expr_base::~expr_base() = default;

expr_base::expr_base() = default;
expr_base::expr_base(sc_data_type_t type) : dtype_(type) {}
expr_base::expr_base(sc_expr_type exp_type) : node_type_(exp_type) {}
expr_base::expr_base(sc_data_type_t type, sc_expr_type exp_type)
    : dtype_(type), node_type_(exp_type) {}

expr::node_ptr(float v) : parent(builder::make_constant(v)) {}
expr::node_ptr(int32_t v) : parent(builder::make_constant(v)) {}
expr::node_ptr(uint64_t v) : parent(builder::make_constant(v)) {}
expr::node_ptr(bool v)
    : node_ptr(make_expr<constant_node>(uint64_t(v), datatypes::boolean)) {}

expr_c::node_ptr(float v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(int32_t v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(uint64_t v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(bool v)
    : node_ptr(make_expr<constant_node>(uint64_t(v), datatypes::boolean)) {}

expr::lvalue_proxy_t::lvalue_proxy_t() : require_remake_(true) {}

expr::lvalue_proxy_t::lvalue_proxy_t(expr data, bool require_remake)
    : data_(std::move(data)), require_remake_(require_remake) {}

expr expr::lvalue_proxy_t::get() const {
    if (require_remake_) {
        return data_->remake();
    } else {
        return expr(data_);
    }
}

expr::lvalue_proxy_t::operator expr() const {
    return get();
}

expr::lvalue_proxy_t::operator expr_c() const {
    return get();
}

void expr::lvalue_proxy_t::operator=(const expr &other) const {
    builder::get_current_builder()->push_assign(get(), other);
}

void expr::lvalue_proxy_t::operator=(expr::lvalue_proxy_t &other) const {
    this->operator=(other.get());
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](
        const std::vector<expr> &index) const {
    return expr::lvalue_proxy_t(builder::make_indexing(*this, index), true);
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](expr index) const {
    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, std::move(index)), true);
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](
        const span_t &index) const {
    return expr::lvalue_proxy_t(get()[index], true);
}

expr::lvalue_proxy_t::lvalue_proxy_t(expr::lvalue_proxy_t &&other)
    : data_(std::move(other.data_)), require_remake_(other.require_remake_) {}

expr::lvalue_proxy_t::lvalue_proxy_t(const expr::lvalue_proxy_t &other)
        = default;

expr::lvalue_proxy_t expr::operator[](const std::vector<expr> &index) const {
    return expr::lvalue_proxy_t(builder::make_indexing(*this, index), true);
}

expr::lvalue_proxy_t expr::operator[](expr index) {
    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, std::move(index)), true);
}

expr::lvalue_proxy_t expr::operator[](const span_t &index) const {
    std::vector<expr> idx;
    idx.reserve(index.index_.size());
    for (auto &i : index.index_) {
        idx.emplace_back(i);
    }

    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, idx, index.length_), true);
}

void print_indents(ostream &os, int indent) {
    for (int i = 0; i < indent; i++) {
        os << "  ";
    }
}

ostream &operator<<(ostream &os, const expr_c &e) {
    return os << e.get();
}

ostream &operator<<(ostream &os, const expr_base *e) {
    e->to_string(os);
    return os;
}

bool expr_base::equals(expr_c other) const {
    ir_comparer cmper;
    return this->equals(std::move(other), cmper);
}

void constant_node::to_string(ostream &os) const {
    if (is_vector()) { os << '('; }
    for (unsigned i = 0; i < value_.size(); i++) {
        switch (dtype_.type_code_) {
            case sc_data_etype::F32: {
                if (this->value_[i].f32 - static_cast<int>(this->value_[i].f32)
                        == 0) {
                    os << this->value_[i].f32 << ".f";
                } else {
                    os.precision(std::numeric_limits<float>::max_digits10);
                    os << this->value_[i].f32;
                }
                break;
            }
            case sc_data_etype::S8:
            case sc_data_etype::S32: os << this->value_[i].s64; break;
            case sc_data_etype::U8:
            case sc_data_etype::U16:
            case sc_data_etype::U32:
            case sc_data_etype::BF16:
            case sc_data_etype::INDEX: os << this->value_[i].u64 << "UL"; break;
            case sc_data_etype::BOOLEAN:
                os << (this->value_[0].u64 ? "true" : "false");
                break;
            case sc_data_etype::POINTER:
                os << "((void*)" << this->value_[i].u64 << ')';
                break;
            default: assert(0 && "Unknown type for const");
        }
        if (i != value_.size() - 1) { os << ',' << ' '; }
    }
    if (is_vector()) { os << ')'; }
}

expr constant_node::remake() const {
    return copy_attr(*this, make_expr<constant_node>(value_, dtype_));
}

#define ASCAST_OR_RETURN(v, other) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    expr_base>; \
    if (!(v).isa<self>()) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    if ((v)->dtype_ != dtype_) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    auto other = v.static_as<self>(); // NOLINT(bugprone-macro-parentheses)

#define DYNCAST_OR_RETURN(v, other) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    expr_base>; \
    if (!(v).instanceof <self>()) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    if ((v)->dtype_ != dtype_) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    auto other = v.static_as<self>(); // NOLINT(bugprone-macro-parentheses)

#define RETURN(val) return ctx.set_result(node_ptr_from_this(), v, (val));

bool constant_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (other->value_.size() != value_.size()) return false;
    switch (dtype_.type_code_) {
        case sc_data_etype::F16:
        case sc_data_etype::BF16:
        case sc_data_etype::F32:
            for (unsigned i = 0; i < value_.size(); i++) {
                if (other->value_[i].f32 != value_[i].f32) RETURN(false);
            }
            RETURN(true);
        case sc_data_etype::POINTER:
        case sc_data_etype::S32:
        case sc_data_etype::U8:
        case sc_data_etype::U16:
        case sc_data_etype::U32:
        case sc_data_etype::S8:
        case sc_data_etype::INDEX:
        case sc_data_etype::BOOLEAN:
            for (unsigned i = 0; i < value_.size(); i++) {
                if (other->value_[i].s64 != value_[i].s64) RETURN(false);
            }
            RETURN(true);
        default: assert(0 && "Unknown type for const");
    }
    return false;
}

void var_node::to_string(ostream &os) const {
    os << name_;
}

expr var_node::remake() const {
    return copy_attr(*this, builder::make_var(dtype_, name_));
}

bool var_node::equals(expr_c v, ir_comparer &ctx) const {
    if (ctx.cmp_var_ref_) {
        if (ctx.get_expr_mapping(node_ptr_from_this(), v)) { return true; }
        RETURN(v.get() == this);
    }
    ASCAST_OR_RETURN(v, other);
    bool name_checking_passed = !ctx.cmp_names_ || (name_ == other->name_);
    if (!name_checking_passed
            || !ctx.check_or_set_expr_mapping(node_ptr_from_this(), v)) {
        RETURN(false);
    }
    // all other checks are done in ASCAST_OR_RETURN
    return true;
}

void cast_node::to_string(ostream &os) const {
    os << dtype_ << '(' << in_ << ')';
}

expr cast_node::remake() const {
    return copy_attr(*this, builder::make_cast(dtype_, in_));
}

bool cast_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return in_->equals(other->in_, ctx);
}

bool binary_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    return l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
}

bool logic_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    return l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
}

bool cmp_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    return l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
}

#define GEN_BINARY(CLASS, OP) \
    void CLASS##_node::to_string(ostream &os) const { \
        os << '(' << l_ << (OP) << r_ << ')'; \
    } \
    expr CLASS##_node::remake() const { \
        return copy_attr(*this, builder::make_##CLASS(l_, r_)); \
    }

void add_node::to_string(ostream &os) const {
    os << '(' << l_ << " + " << r_ << ')';
}
expr add_node::remake() const {
    auto ret = builder::make_add(l_, r_);
    if (dtype_ != datatypes::undef) ret->dtype_ = dtype_;
    return copy_attr(*this, std::move(ret));
}

GEN_BINARY(sub, " - ")
GEN_BINARY(mul, " * ")
GEN_BINARY(div, " / ")
GEN_BINARY(mod, " % ")
GEN_BINARY(cmp_eq, " == ")
GEN_BINARY(cmp_lt, " < ")
GEN_BINARY(cmp_le, " <= ")
GEN_BINARY(cmp_gt, " > ")
GEN_BINARY(cmp_ge, " >= ")
GEN_BINARY(cmp_ne, " != ")
GEN_BINARY(logic_and, " && ")
GEN_BINARY(logic_or, " || ")

void logic_not_node::to_string(ostream &os) const {
    os << "!(" << in_ << ')';
}

bool logic_not_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return in_->equals(other->in_, ctx);
}

expr logic_not_node::remake() const {
    return copy_attr(*this, builder::make_logic_not(in_));
}

void select_node::to_string(ostream &os) const {
    os << "(" << cond_ << "?" << l_ << ":" << r_ << ")";
}

bool select_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return cond_->equals(other->cond_, ctx) && l_->equals(other->l_, ctx)
            && r_->equals(other->r_, ctx);
}

expr select_node::remake() const {
    return copy_attr(*this, builder::make_select(cond_, l_, r_));
}

void indexing_node::to_string(ostream &os) const {
    os << ptr_ << '[';
    assert(!idx_.empty());
    for (size_t i = 0; i < idx_.size() - 1; i++) {
        os << idx_.at(i) << ", ";
    }
    os << idx_.back();
    if (dtype_.lanes_ > 1) { os << " @ " << dtype_.lanes_; }
    if (mask_.defined()) { os << " M= " << mask_; }
    os << ']';
}

expr indexing_node::remake() const {
    return copy_attr(
            *this, builder::make_indexing(ptr_, idx_, dtype_.lanes_, mask_));
}

bool indexing_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return ptr_->equals(other->ptr_, ctx)
            && ctx.set_result(node_ptr_from_this(), v,
                    ctx.expr_arr_equals(idx_, other->idx_))
            && ctx.check_equals_may_null(mask_, other->mask_);
}

call_node::call_node(const func_t &func, const std::vector<expr> &args)
    : call_node(func, args, {}) {}

call_node::call_node(const func_t &func, const std::vector<expr> &args,
        std::vector<parallel_attr_t> &&para_attr)
    : expr_base(func->ret_type_, sc_expr_type::call)
    , func_(func)
    , args_(args)
    , para_attr_(std::move(para_attr)) {}

call_node::parallel_attr_t::parallel_attr_t(expr begin_, expr end_, expr step_)
    : begin_(std::move(begin_))
    , end_(std::move(end_))
    , step_(std::move(step_)) {}

void call_node::to_string(ostream &os) const {
    os << func_->name_ << '(';
    if (!args_.empty()) {
        for (unsigned i = 0; i < args_.size() - 1; i++) {
            os << args_.at(i) << ", ";
        }
        os << args_.back();
    }
    os << ')';
    if (!para_attr_.empty()) {
        os << "@parallel(";
        for (auto &v : para_attr_) {
            os << '[' << v.begin_ << ", " << v.end_ << ", " << v.step_ << "], ";
        }
        os << ')';
    }
}

expr call_node::remake() const {
    return copy_attr(*this, builder::make_call(func_, args_));
}

// for the callee, just check if pointer is same
bool call_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    auto shared = node_ptr_from_this();
    if (ctx.cmp_callee_) {
        if (!func_->equals(other->func_, ctx)) return false;
    } else {
        if (!ctx.set_result(shared, v, func_.ptr_same(other->func_)))
            return false;
    }
    if (para_attr_.size() != other->para_attr_.size()) { RETURN(false); }
    for (unsigned i = 0; i < para_attr_.size(); i++) {
        auto &p = para_attr_[i];
        auto &op = other->para_attr_[i];
        if (!p.begin_->equals(op.begin_, ctx) || !p.end_->equals(op.end_, ctx)
                || !p.step_->equals(op.step_, ctx)) {
            return false;
        }
    }
    RETURN(ctx.expr_arr_equals(args_, other->args_));
}

void tensor_node::to_string(ostream &os) const {
    os << name_;
}

void tensor_node::to_string_full(ostream &os) {
    os << name_ << ": [" << elem_dtype_ << " * ";
    if (!dims_.empty()) {
        for (unsigned i = 0; i < dims_.size() - 1; i++) {
            os << dims_.at(i) << " * ";
        }
        os << dims_.back();
    }
    os << ']';
    if (address_space_ != address_space::automatic) {
        switch (address_space_) {
            case address_space::device: os << " device"; break;
            case address_space::host: os << " host"; break;
            default: assert(0); break;
        }
    }
}

expr tensor_node::remake() const {
    return copy_attr(*this,
            builder::make_tensor(
                    name_, dims_, elem_dtype_, address_space_, init_value_));
}

// ignore the names
bool tensor_node::equals(expr_c v, ir_comparer &ctx) const {
    if (ctx.cmp_var_ref_) {
        if (ctx.get_expr_mapping(node_ptr_from_this(), v)) { return true; }
        RETURN(v.get() == this);
    }
    ASCAST_OR_RETURN(v, other);
    bool name_checking_passed = !ctx.cmp_names_ || (name_ == other->name_);
    if (!name_checking_passed || address_space_ != other->address_space_
            || dtype_ != other->dtype_ || elem_dtype_ != other->elem_dtype_
            || !ctx.check_or_set_expr_mapping(node_ptr_from_this(), v)) {
        RETURN(false);
    }
    if (init_value_) {
        if (!other->init_value_) { RETURN(false); }
        if (init_value_->size_ != other->init_value_->size_
                || memcmp(init_value_->data_, other->init_value_->data_,
                        other->init_value_->size_)) {
            RETURN(false);
        }
    } else {
        if (other->init_value_) { RETURN(false); }
    }
    RETURN(ctx.expr_arr_equals(dims_, other->dims_));
}

void tensorptr_node::to_string(ostream &os) const {
    os << '&' << base_;
}

expr tensorptr_node::remake() const {
    return copy_attr(
            *this, make_expr<tensorptr_node>(base_, shape_, is_slice_));
}

bool tensorptr_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (other->is_slice_ != is_slice_) { RETURN(false); }
    if (!ctx.expr_arr_equals(shape_, other->shape_)) { RETURN(false); }
    return base_->equals(other->base_, ctx);
}

void intrin_call_node::to_string(ostream &os) const {
    auto &v = get_intrinsic_handler(type_);
    os << v.name_ << '(';
    if (!args_.empty()) {
        for (unsigned i = 0; i < args_.size() - 1; i++) {
            os << args_.at(i) << ", ";
        }
        os << args_.back();
    }
    os << ')';
}

intrin_call_node::intrin_call_node(intrin_type intrin,
        const std::vector<expr> &args, const any_map_t &attrs)
    : expr_base(sc_expr_type::intrin_call)
    , type_(intrin)
    , args_(args)
    , intrin_attrs_(utils::make_unique<any_map_t>(attrs)) {
    get_intrinsic_handler(type_).on_initialize(*this);
}

expr intrin_call_node::remake() const {
    return copy_attr(
            *this, make_expr<intrin_call_node>(type_, args_, *intrin_attrs_));
}

bool intrin_call_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (type_ != other->type_) { RETURN(false); }
    RETURN(ctx.expr_arr_equals(args_, other->args_));
}

void func_addr_node::to_string(ostream &os) const {
    os << '&' << func_->name_;
}

expr func_addr_node::remake() const {
    return copy_attr(*this, make_expr<func_addr_node>(func_));
}

bool func_addr_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    RETURN(func_ == other->func_);
}

const std::string &get_node_name(const expr &e) {
    tensor t = e.as<tensor>();
    if (t.get() != nullptr) { return t->name_; }

    var v = e.as<var>();
    if (v.get() != nullptr) { return v->name_; }

    COMPILE_ASSERT(
            false, "Not an expr_base subclass that has a 'name_' member.");
}

std::vector<expr> dims_to_expr(const sc_dims &dim) {
    std::vector<sc::expr> dim_expr;
    dim_expr.reserve(dim.size());
    for (auto d : dim) {
        uint64_t unsigned_d = d;
        dim_expr.emplace_back(unsigned_d);
    }
    return dim_expr;
}

std::vector<expr_c> dims_to_expr_c(const sc_dims &dim) {
    std::vector<sc::expr_c> dim_expr;
    dim_expr.reserve(dim.size());
    for (auto d : dim) {
        uint64_t unsigned_d = d;
        dim_expr.emplace_back(unsigned_d);
    }
    return dim_expr;
}

} // namespace sc
