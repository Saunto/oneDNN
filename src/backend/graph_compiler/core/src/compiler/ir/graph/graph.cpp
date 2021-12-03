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

#include <algorithm>
#include <vector>

#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <util/hash_utils.hpp>

namespace sc {

SC_MODULE(sc_graph);

sc_op_ptr op_traits::auto_copyable_t::copy(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ths = dynamic_cast<sc_op *>(this);
    return mgr.make(ths->op_name_, ins, outs, ths->attrs_);
}

std::vector<graph_tensor_ptr> copy_logical_tsr(
        const std::vector<graph_tensor_ptr> &v) {
    std::vector<graph_tensor_ptr> ret;
    ret.reserve(v.size());
    for (auto &t : v) {
        ret.emplace_back(std::make_shared<graph_tensor>(nullptr, t->details_));
    }
    return ret;
}

static std::vector<expr> *get_tensor_dims(const expr &tsr) {
    if (tsr.isa<tensor>()) {
        auto t = tsr.static_as<tensor>();
        return &t->dims_;
    } else {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        return &tsr.static_as<tensorptr>()->shape_;
    }
}

const std::vector<expr> &tensor_slice::get_base_dims() const {
    return *get_tensor_dims(tptr_);
}

sc_data_type_t tensor_slice::get_base_dtype() const {
    return get_real_tensor()->elem_dtype_;
}

tensor tensor_slice::get_real_tensor() const {
    auto &base = tptr_->base_;
    COMPILE_ASSERT(base.isa<indexing>(),
            "tensor_ptr base should be indexing, but got: " << base);
    auto tsr = base->ptr_;
    while (!tsr.isa<tensor>()) {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        auto base = tsr.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        tsr = base.checked_as<indexing>()->ptr_;
    }
    return tsr.static_as<tensor>();
}

slice_range tensor_slice::get_ranges() const {
    COMPILE_ASSERT(get_shape().size() == tptr_->base_->idx_.size(),
            "Unmatched shape and idx found");
    auto shape = get_shape();
    auto offset = tptr_->base_->idx_;
    slice_range ranges;
    for (int64_t i = 0; i < nslice_dims(); i++) {
        ranges.emplace_back(std::make_pair(offset[i], shape[i]));
    }
    return ranges;
}

bool tensor_slice::full_on_axes(const std::vector<int> &axes) const {
    auto &dims = get_base_dims();
    auto &idx = tptr_->base_->idx_;
    auto &shape = get_shape();
    for (auto &ax : axes) {
        if (!idx[ax].isa<constant>() || !shape[ax].isa<constant>()) {
            return false;
        }
        if (get_const_as_int(idx[ax].checked_as<constant>()) != 0
                || get_const_as_int(shape[ax].checked_as<constant>())
                        != get_const_as_int(dims[ax].checked_as<constant>())) {
            return false;
        }
    }
    return true;
}

bool tensor_slice::is_full() const {
    auto &dims = get_base_dims();
    std::vector<int> total_axes;
    total_axes.reserve(static_cast<int>(dims.size()));
    for (int i = 0; i < static_cast<int>(dims.size()); i++) {
        total_axes.emplace_back(i);
    }
    return full_on_axes(total_axes);
}

tensor_slice::tensor_slice(const expr &tsr) {
    if (tsr.isa<tensor>()) {
        auto t = tsr.static_as<tensor>();
        tptr_ = builder::tensor_ptr(
                tsr, std::vector<expr>(t->dims_.size(), 0), {}, true)
                        .static_as<tensorptr>();
        shape_ = t->dims_;
    } else {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        tptr_ = tsr.static_as<tensorptr>();
        shape_ = tptr_->shape_;
    }
}

tensor_slice::tensor_slice(const expr &tsr, slice_range &&range) {
    auto dims = get_tensor_dims(tsr);
    if (dims->size() != range.size())
        COMPILE_ASSERT(dims->size() == 1
                        && get_const_as_int((*dims)[0].checked_as<constant>())
                                == 1,
                "Unexpected range found. Tensor:"
                        << (tsr.isa<tensor>() ? tsr.static_as<tensor>()->name_
                                              : "")
                        << " have dims: " << utils::print_vector(*dims)
                        << " but got range size: " << range.size());
    tptr_ = builder::tensor_ptr(tsr,
            dims->size() != range.size() ? std::vector<expr> {0}
                                         : get_slice_idx(range),
            {}, true)
                    .static_as<tensorptr>();
    shape_ = get_slice_shape(range);
}

graph_tensor::graph_tensor(sc_op *owner) : producer_owner_(owner) {}
graph_tensor::graph_tensor(sc_op *owner, const logical_tensor_t &lt)
    : details_(lt), producer_owner_(owner) {}

graph_tensor::graph_tensor(sc_op *owner, const sc_data_format_t &format,
        const sc_dims &plain_shape, const sc_data_type_t &type)
    : details_(format, plain_shape, type), producer_owner_(owner) {}

const sc_dims &logical_tensor_t::get_blocking_dims() const {
    return dims_;
}

void logical_tensor_t::internal_update() {
    dims_ = sc_data_format_t::get_blocking_shapes(plain_dims_, format_);
}

// sets the logical dims in plain format
void logical_tensor_t::set_plain_dims(const sc_dims &plain_dims) {
    plain_dims_ = plain_dims;
    internal_update();
}

// TODO(xxx): this logic maybe not correct, just distinguish with set_plain_dims
void logical_tensor_t::set_blocking_dims(const sc_dims &blocking_dims) {
    // assert(format_.format_code_ == format_kinds::any);
    format_.format_code_ = format_kinds::any;
    plain_dims_ = blocking_dims;
    dims_ = blocking_dims;
}

void logical_tensor_t::set_format(const sc_data_format_t &newv) {
    format_ = newv;
    internal_update();
}

void graph_tensor::attach_use(sc_op_ptr op, int index) {
    uses_.emplace_back(std::make_pair(index, std::move(op)));
}

void graph_tensor::detach_use(const sc_op_ptr &op) {
    for (auto itr = uses_.begin(); itr != uses_.end();) {
        if (itr->second == op) {
            itr = uses_.erase(itr);
        } else {
            ++itr;
        }
    }
}

void graph_tensor::detach_use(const sc_op_ptr &op, int input_idx) {
    for (auto itr = uses_.begin(); itr != uses_.end();) {
        if (itr->first == input_idx && itr->second == op) {
            itr = uses_.erase(itr);
        } else {
            ++itr;
        }
    }
}

void graph_tensor::replace_with(const graph_tensor_ptr &v) {
    while (!uses_.empty()) {
        auto node = uses_.front();
        node.second->replace_input(node.first, v);
    }
}

graph_tensor_ptr graph_tensor::copy() {
    return std::make_shared<graph_tensor>(producer_owner_, details_);
}

void sc_op::replace_input(size_t index, const graph_tensor_ptr &new_input) {
    assert(index < info_.inputs_.size());
    assert(info_.inputs_[index]->details_.get_plain_dims()
            == new_input->details_.get_plain_dims());
    info_.inputs_[index]->detach_use(shared_from_this(), index);
    info_.inputs_[index] = new_input;
    new_input->attach_use(shared_from_this(), index);
}

void sc_op::replace_uses_with_and_remove(const sc_op_ptr &replacer) {
    assert(info_.outputs_.size() == replacer->info_.outputs_.size());
    for (unsigned i = 0; i < info_.outputs_.size(); i++) {
        auto &ths_out = info_.outputs_[i];
        auto &replace_out = replacer->info_.outputs_[i];
        ths_out->replace_with(replace_out);
    }
    remove();
}

void sc_op::remove() {
    for (auto &in : info_.inputs_) {
        in->detach_use(shared_from_this());
    }
    info_.inputs_.clear();
    info_.outputs_.clear();
    is_removed_ = true;
}

// template op and fusible op common constructor
sc_op::sc_op(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &producer_lt,
        const std::vector<graph_tensor_ptr> &consumer_lt,
        const any_map_t &attrs)
    : attrs_(attrs), op_name_(op_name) {
    info_.inputs_ = producer_lt;
    info_.outputs_ = consumer_lt;
    for (auto &op : info_.outputs_) {
        op->producer_owner_ = this;
    }
}

void sc_graph_t::reset_op_ids() {
    for (auto it = ops_.begin(); it != ops_.end();) {
        if ((*it)->is_removed_
                || ((*it)->get_inputs().empty()
                        && (*it)->get_outputs().empty())) {
            it = ops_.erase(it);
        } else {
            ++it;
        }
    }
    for (size_t i = 0; i < ops_.size(); ++i) {
        ops_[i]->logical_op_id_ = i;
    }
}

float sc_graph_t::get_gflop() const {
    float gflop = 0.f;
    for (auto &op : ops_) {
        if (auto tune_op = op->dyn_cast<tunable_op_t>()) {
            if (op->is_removed_) continue;
            gflop += tune_op->get_gflop();
        }
    }
    return gflop;
}

void sc_graph_t::add(const sc_op_ptr &ret) {
    assert(ret->logical_op_id_ == 0);
    ret->logical_op_id_ = ops_.size();
    for (auto &outs : ret->info_.outputs_) {
        assert(outs->producer_owner_ == nullptr
                || outs->producer_owner_ == ret.get());
        outs->producer_owner_ = ret.get();
    }
    for (unsigned i = 0; i < ret->info_.inputs_.size(); i++) {
        ret->info_.inputs_[i]->attach_use(ret, i);
    }
    ops_.emplace_back(ret);
}

std::shared_ptr<sc_op> sc_graph_t::make(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &inputs,
        const std::vector<graph_tensor_ptr> &outputs, const any_map_t &attrs) {
    std::shared_ptr<sc_op> ret, in_ret;
    // internally create input_op
    // todo: LLGA-sc front end should create input_op first, instead of creating
    // it internally
    for (auto &ins : inputs) {
        if (!ins->producer_owner_) {
            auto in_ret = std::make_shared<input_op>(
                    std::vector<graph_tensor_ptr> {ins});
            in_ret->logical_op_id_ = ops_.size();
            ops_.emplace_back(std::move(in_ret));
        }
    }

    std::string decay_op_name = graph::decay_quantized_op_name(op_name);
    // firstly search template, secondly search fusible
    // todo: add all tunable ops
    if (auto f = get_op_factory(decay_op_name)) {
        ret = f(inputs, outputs, attrs);
    } else {
        COMPILE_ASSERT(false, "Unsupported op: " << decay_op_name);
    }
    bool is_quantized = utils::string_startswith(op_name, "quantized");
    if (is_quantized) {
        ret->dyn_cast<op_traits::may_quantize_t>()->is_quantized_ = true;
    }
    add(ret);
    return ret;
}

std::shared_ptr<sc_op> sc_graph_t::make_output(
        const std::vector<graph_tensor_ptr> &inputs, const any_map_t &attrs) {
    auto ret = std::make_shared<output_op>(inputs);
    ret->attrs_ = attrs;
    for (unsigned i = 0; i < inputs.size(); i++) {
        inputs[i]->attach_use(ret, i);
    }
    ret->logical_op_id_ = ops_.size();
    ops_.emplace_back(ret);
    return ret;
}

std::shared_ptr<sc_op> sc_graph_t::make_input(
        const std::vector<graph_tensor_ptr> &inputs, const any_map_t &attrs) {
    auto ret = std::make_shared<input_op>(inputs);
    ret->attrs_ = attrs;
    ret->logical_op_id_ = ops_.size();
    ops_.emplace_back(ret);
    return ret;
}

std::vector<sc_op_ptr> sc_graph_t::get_output_ops() {
    std::vector<sc_op_ptr> output_ops;
    for (auto &op : ops_) {
        if (op->isa<output_op>()) { output_ops.push_back(op); }
    }
    return output_ops;
}
std::vector<sc_op_ptr> sc_graph_t::get_input_ops() {
    std::vector<sc_op_ptr> input_ops;
    for (auto &op : ops_) {
        if (op->isa<input_op>()) { input_ops.push_back(op); }
    }
    return input_ops;
}

std::vector<sc_op_ptr> sc_graph_t::get_input_or_const_ops() const {
    std::vector<sc_op_ptr> input_ops;
    for (auto &op : ops_) {
        if (op->isa<input_op>() || op->isa<constant_op_t>()) {
            input_ops.push_back(op);
        }
    }
    return input_ops;
}

bool sc_op::compare_contents(const sc_op *other) const {
    if (op_name_ != other->op_name_) { return false; }
    int numattrs = 0, othernumattrs = 0;
    auto &othermap = other->attrs_.as_map();
    for (auto &kv : attrs_.as_map()) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        numattrs++;
        auto otherkv = othermap.find(kv.first);
        if (otherkv == othermap.end()) { return false; }
        if (kv.second.cmp(otherkv->second) != 0) { return false; }
    }
    for (auto &kv : othermap) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        othernumattrs++;
    }
    if (numattrs != othernumattrs) { return false; }

    return true;
}

size_t sc_op::hash_contents() const {
    size_t seed = 0;
    hash_combine(seed, this->op_name_);
    for (auto &kv : attrs_.as_map()) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        // To hash unordered_map, use `XOR`, which satisfies commutative law.
        // Otherwise, for ordered containers (like arrays), use `hash_combine`
        // to distinguish result from the differnt sequence order.
        seed ^= kv.second.hash();
    }
    return seed;
}

static std::unordered_map<std::string, op_factory_func> &get_op_factory_map() {
    static std::unordered_map<std::string, op_factory_func> op_map;
    return op_map;
}

op_factory_func get_op_factory(const std::string &name) {
    auto &op_map = get_op_factory_map();
    auto itr = op_map.find(name);
    if (itr != op_map.end()) { return itr->second; }
    return nullptr;
}

void set_op_factory(const std::string &name, op_factory_func f) {
    auto &op_map = get_op_factory_map();
    COMPILE_ASSERT(op_map.find(name) == op_map.end(),
            "The op has already registered!");
    op_map[name] = f;
}

float sc_op::get_gflop() {
    return 0.0f;
}

namespace graph {
std::string decay_quantized_op_name(const std::string &op_name) {
    bool is_quantized = utils::string_startswith(op_name, "quantized");
    std::string qstring = "quantized_";
    std::string decay_op_name = is_quantized
            ? op_name.substr(qstring.size(), op_name.size() - qstring.size())
            : op_name;
    return decay_op_name;
}

void get_logical_tensors(
        ltensors *ins, const std::vector<graph_tensor_ptr> &flts) {
    ins->reserve(flts.size());
    for (auto &in : flts) {
        ins->emplace_back(in->details_);
    }
}

expr tensor_detail_to_ir_tensor(
        const std::string &name, const logical_tensor_t &tsrd) {
    return builder::make_tensor(
            name, dims_to_expr(tsrd.get_blocking_dims()), tsrd.dtype_);
}

std::vector<expr> tensor_detail_to_ir_tensor(const std::string &name_prefix,
        const std::vector<logical_tensor_t> &tsrs) {
    std::vector<expr> ret;
    ret.reserve(tsrs.size());
    for (size_t i = 0; i < tsrs.size(); i++) {
        ret.emplace_back(tensor_detail_to_ir_tensor(
                name_prefix + std::to_string(i), tsrs[i]));
    }
    return ret;
}

std::vector<expr> tensor_detail_to_ir_tensor(const std::string &name_prefix,
        const std::vector<graph_tensor_ptr> &tsrs) {
    std::vector<expr> ret;
    ret.reserve(tsrs.size());
    for (size_t i = 0; i < tsrs.size(); i++) {
        ret.emplace_back(tensor_detail_to_ir_tensor(
                name_prefix + std::to_string(i), tsrs[i]->details_));
    }
    return ret;
}

ltensors extract_detail_from_tensors(
        const std::vector<std::shared_ptr<graph_tensor>> &flts) {
    std::vector<logical_tensor_t> ret;
    ret.reserve(flts.size());
    for (auto &in : flts) {
        ret.emplace_back(in->details_);
    }
    return ret;
}

} // namespace graph
} // namespace sc

namespace std {
std::size_t hash<sc::logical_tensor_t>::operator()(
        const sc::logical_tensor_t &k) const {
    size_t seed = 0;
    hash_combine(seed, k.dtype_);
    hash_combine(seed, k.format_);
    for (size_t i = 0; i < k.plain_dims_.size(); i++) {
        hash_combine(seed, k.plain_dims_[i]);
    }
    return seed;
}
} // namespace std
