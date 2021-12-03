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
#include "tensor_shrink.hpp"
#include <utility>
#include <vector>
#include "../ir_comparer.hpp"
#include "../visitor.hpp"
#include <common/dimensions.hpp>
#include <compiler/ir/builder.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace sc {

// check tensor node attr
static bool is_tensor_and_should_shrink(const expr &e) {
    return e.isa<tensor>() && e->attr_
            && e->attr_->has_key(tensor_shrinker_attrs::should_shrink);
}
// check tensorptr node attr
static bool is_tensorptr_and_should_shrink(const expr &e) {
    return e.isa<tensorptr>() && e.static_as<tensorptr>()->base_.defined()
            && e.static_as<tensorptr>()->base_->ptr_.isa<tensor>()
            && is_tensor_and_should_shrink(
                    e.static_as<tensorptr>()->base_->ptr_.static_as<tensor>());
}
static bool is_reshaped_and_should_shrink(const expr &e) {
    return e.isa<tensorptr>() && !e.static_as<tensorptr>()->is_slice_
            && e.static_as<tensorptr>()->base_.defined()
            && e.static_as<tensorptr>()->base_->ptr_.isa<tensor>() && e->attr_
            && e->attr_->has_key(tensor_shrinker_attrs::should_shrink);
}

static constexpr const char *temp_shrink_tag = "tensor_shrinker.def";

/**
 * Due to in some cases, the brgemm may access discontinuous memory. If applied
 * tensor shrink, it should dynamically change leading dimension argument named
 * `LDX` in brgemm args list. E.g. For output[A,C,B,D], brgemm will write back
 * it partial result into [1,C,1,D] with LDC = B*D. However, the local buffer
 * should be shrinked into [C,D], but with new LDC = D.
 * */
bool check_brgemm_LDX(
        std::vector<expr> &args, const expr &buffer, int LD_arg_idx) {
    COMPILE_ASSERT(static_cast<uint64_t>(LD_arg_idx) < args.size(),
            "arg idx should not exceed args length, but got "
                    << args.size() << " args with LDX idx: " << LD_arg_idx);
    COMPILE_ASSERT(buffer.isa<tensor>() || buffer.isa<tensorptr>(),
            "tensor or tensorptr is expected for the buffer of brgemm");
    if (is_tensor_and_should_shrink(buffer)
            || is_tensorptr_and_should_shrink(buffer)) {
        auto tsr = buffer.isa<tensor>()
                ? buffer.static_as<tensor>()
                : buffer.static_as<tensorptr>()
                          ->base_->ptr_.static_as<tensor>();
        auto &shrink_info = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                tensor_shrinker_attrs::should_shrink);
        COMPILE_ASSERT(shrink_info.shape_.size() == tsr->dims_.size(),
                "Bad number of dimensions for indexing access");
        COMPILE_ASSERT(args[LD_arg_idx].isa<constant>(),
                "Constant LDX is expected, but got " << args[LD_arg_idx]);
        int64_t LDX = get_expr_as_int(args[LD_arg_idx]);
        int64_t acc_orig = 1, acc_shrink = 1;
        for (int64_t i = static_cast<int64_t>(shrink_info.shape_.size()) - 1;
                i >= 0; i--) {
            if (acc_orig == LDX) {
                if (acc_shrink == acc_orig)
                    return false;
                else {
                    args[LD_arg_idx] = make_expr<constant_node>(
                            acc_shrink, datatypes::s32);
                    return true;
                }
            }
            acc_orig *= get_expr_as_int(tsr->dims_[i]);
            acc_shrink *= get_expr_as_int(shrink_info.shape_[i]);
        }
        COMPILE_ASSERT(0,
                "Unexpected LDX found: " << LDX
                                         << " for corresponding tensor dims: "
                                         << utils::print_vector(tsr->dims_));
    }
    return false;
}

class shrinker_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;

    std::unordered_map<expr, expr> replace_map;

    stmt_c visit(define_c v) override {
        if (is_tensor_and_should_shrink(v->var_)) {
            auto tsr = v->var_.static_as<tensor>();
            COMPILE_ASSERT(!tsr->init_value_ && !v->init_.defined()
                            && v->linkage_ == linkage::local,
                    "The tensor to shrink should not have init value or be "
                    "re-scheduled. And it should be a local tensor: "
                            << v);
            auto &shrink_info
                    = v->var_->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            COMPILE_ASSERT(shrink_info.shape_.size() == tsr->dims_.size(),
                    "Bad shape for shrinking the tensor: "
                            << v << ", target shape = "
                            << utils::print_vector(shrink_info.shape_));
            auto replacer = copy_attr(*tsr,
                    builder::make_tensor(tsr->name_ + "_shr",
                            shrink_info.shape_, tsr->elem_dtype_,
                            tsr->address_space_));
            replacer->attr_->remove(tensor_shrinker_attrs::should_shrink);
            replace_map[tsr] = replacer;
            auto ret = builder::make_var_tensor_def_unattached(
                    replacer, v->linkage_);
            // if the tensor definition is moved
            if (shrink_info.move_def_.defined()) {
                shrink_info.move_def_->attr()[temp_shrink_tag] = ret;
                // the tensor def is moved, return empty
                return builder::make_stmts_unattached({});
            }
            return ret;
        }
        return ir_visitor_t::visit(v);
    }

    expr_c visit(intrin_call_c v) override {
        auto intrin = ir_visitor_t::visit(v).checked_as<intrin_call_c>();
        if (v->type_ == intrin_type::brgemm) {
            // new args
            auto args_cpy = intrin->args_;
            std::vector<std::pair<int, int>> check_LDX_list = {
                    // // Input fusion
                    // {brgemm_args::A, brgemm_args::LDA},
                    // {brgemm_args::B, brgemm_args::LDB},
                    // Output fusion
                    {brgemm_args::C, brgemm_args::LDC},
            };
            bool changed = false;
            for (auto &check_pair : check_LDX_list) {
                // need to check old args v, due to some of `attr` maybe removed
                // in new args.
                if (check_brgemm_LDX(args_cpy, v->args_[check_pair.first],
                            check_pair.second)) {
                    changed = true;
                }
            }
            if (changed) {
                return copy_attr(*intrin,
                        make_expr<intrin_call_node>(intrin_type::brgemm,
                                args_cpy, *intrin->intrin_attrs_));
            }
        }
        // TODO(xxx): add implement for list brgemm, if necessary
        // else if (v->type_ == intrin_type::list_brgemm) {}
        return intrin;
    }

    expr_c visit(tensor_c v) override {
        // if shrinked tensors should not go here, unless it is a direct use of
        // the tensor, instead of indexing
        COMPILE_ASSERT(!v->attr_
                        || !v->attr_->has_key(
                                tensor_shrinker_attrs::should_shrink),
                "The shrinked tensor is referenced without indexing: " << v);
        return ir_visitor_t::visit(std::move(v));
    }

    expr_c visit(indexing_c v) override {
        if (is_tensor_and_should_shrink(v->ptr_)) {
            auto tsr = v->ptr_.static_as<tensor>();
            COMPILE_ASSERT(v->idx_.size() == tsr->dims_.size(),
                    "Bad number of dimensions for indexing access");
            auto itr = replace_map.find(tsr);
            COMPILE_ASSERT(itr != replace_map.end(),
                    "Tensor used before definition: " << v);
            std::vector<expr> new_idx;
            bool changed = ir_visitor_t::dispatch_expr_vector(v->idx_, new_idx);
            auto &shrink_info
                    = tsr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            // already checked that in visit(define_c v). using assert now
            assert(new_idx.size() == shrink_info.base_.size());
            for (size_t i = 0; i < new_idx.size(); i++) {
                new_idx[i] = new_idx[i] - shrink_info.base_[i];
            }
            return builder::make_indexing(
                    itr->second, new_idx, v->dtype_.lanes_, v->mask_);
        }
        return ir_visitor_t::visit(v);
    }

    /**
     * TO deal with reshaped tensor, we need to transform both idx and shape
     * from `tensorptr(tensorptr(base,{0,..},shape,false),idx,{},true)` to
     * `tensorptr(tensorptr(base,{0,..},newshape,false),newidx,{},true)`
     * */
    expr_c visit(tensorptr_c v) override {
        // transform based reshaped tensor's shape
        if (is_reshaped_and_should_shrink(v.remove_const())) {
            auto tptr = ir_visitor_t::visit(v).checked_as<tensorptr>();
            auto &shrink_info
                    = tptr->attr_->get<tensor_shrinker_t::shrink_info_t>(
                            tensor_shrinker_attrs::should_shrink);
            return builder::tensor_ptr(tptr->base_->ptr_,
                    std::vector<expr>(tptr->base_->idx_.size(), expr(0)),
                    shrink_info.shape_, v->is_slice_);
        }
        // transform reshaped tensorptr's idx
        else if (v->base_->ptr_.isa<tensorptr>()
                && is_reshaped_and_should_shrink(
                        v->base_->ptr_.static_as<tensorptr>())) {
            // get shrink info firstly due to it will not be returned by visit
            // below
            auto &shrink_info
                    = v->base_->ptr_->attr_
                              ->get<tensor_shrinker_t::shrink_info_t>(
                                      tensor_shrinker_attrs::should_shrink);
            auto tptr = ir_visitor_t::visit(v).checked_as<tensorptr>();
            auto inner_tptr = tptr->base_->ptr_;
            std::vector<expr> newidx;
            bool changed = ir_visitor_t::dispatch_expr_vector(
                    tptr->base_->idx_, newidx);

            // already checked that in visit(define_c v). using assert now
            assert(newidx.size() == shrink_info.base_.size());
            for (size_t i = 0; i < newidx.size(); i++) {
                newidx[i] = newidx[i] - shrink_info.base_[i];
            }
            return builder::tensor_ptr(tptr->base_->ptr_, newidx, {}, true);
        }
        return ir_visitor_t::visit(v);
    }

    stmt_c visit(stmts_c s) override {
        if (s->attr_ && s->attr_->has_key(temp_shrink_tag)) {
            COMPILE_ASSERT(
                    s->seq_.empty(), "Shrink definition placeholder not empty");
            auto def = s->attr_->get<stmt>(temp_shrink_tag);
            s->attr_->as_map().clear();
            return def;
        }
        return ir_visitor_t::visit(std::move(s));
    }
};

func_c tensor_shrinker_t::operator()(func_c f) {
    shrinker_impl_t impl;
    return impl.dispatch(f);
}
stmt_c tensor_shrinker_t::operator()(stmt_c f) {
    shrinker_impl_t impl;
    return impl.dispatch(std::move(f));
}

} // namespace sc
