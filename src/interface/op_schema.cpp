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

#include <limits>
#include <memory>

#include "interface/op_schema.hpp"
#include "interface/opset.hpp"
#include "utils/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {

op_schema_t::op_schema_t() : op_kind_(op_kind::LastSymbol), version_(0) {}

op_schema_t::op_schema_t(op_kind_t kind, opset_version version)
    : op_kind_(kind), version_(version) {}

// the rvalue reference design is based on the fact that these
// functions are only called internally with rvalue intputs.
op_schema_t &op_schema_t::set_op_kind(op_kind_t kind) {
    op_kind_ = kind;
    return *this;
}

op_kind_t op_schema_t::get_op_kind() const {
    return op_kind_;
}

op_schema_t &op_schema_t::set_doc(std::string &&doc) {
    doc_ = std::move(doc);
    return *this;
}

const std::string &op_schema_t::get_doc() const {
    return doc_;
}

op_schema_t &op_schema_t::since_version(opset_version n) {
    version_ = n;
    return *this;
}

opset_version op_schema_t::get_since_version() const {
    return version_;
}

op_schema_t &op_schema_t::set_num_inputs(std::set<size_t> &&input_num) {
    num_inputs_ = std::move(input_num);
    return *this;
}

op_schema_t &op_schema_t::set_num_inputs(size_t input_num) {
    num_inputs_.insert(input_num);
    return *this;
}

std::set<size_t> op_schema_t::get_num_inputs() const {
    return num_inputs_;
}

op_schema_t &op_schema_t::set_num_outputs(std::set<size_t> &&output_num) {
    num_outputs_ = std::move(output_num);
    return *this;
}

op_schema_t &op_schema_t::set_num_outputs(size_t output_num) {
    num_outputs_.insert(output_num);
    return *this;
}

std::set<size_t> op_schema_t::get_num_outputs() const {
    return num_outputs_;
}

op_schema_t &op_schema_t::set_input(size_t in_offset, std::string &&in_name,
        std::string &&in_description, data_type_t dtype) {
    verify_input_(in_offset);
    inputs_.emplace_back(op_parameter_t(
            std::move(in_name), std::move(in_description), dtype));

    return *this;
}

op_schema_t &op_schema_t::set_input(size_t in_offset, std::string &&in_name,
        std::string &&in_description, std::set<data_type_t> &&dtypes) {
    verify_input_(in_offset);
    inputs_.emplace_back(op_parameter_t(
            std::move(in_name), std::move(in_description), std::move(dtypes)));

    return *this;
}

const std::vector<op_schema_t::op_parameter_t> &
op_schema_t::get_inputs() const {
    return inputs_;
}

op_schema_t &op_schema_t::set_output(size_t out_offset, std::string &&out_name,
        std::string &&out_description, data_type_t dtype) {
    verify_output_(out_offset);
    outputs_.emplace_back(op_parameter_t(
            std::move(out_name), std::move(out_description), dtype));

    return *this;
}

op_schema_t &op_schema_t::set_output(size_t out_offset, std::string &&out_name,
        std::string &&out_description, std::set<data_type_t> &&dtype) {
    verify_output_(out_offset);
    outputs_.emplace_back(op_parameter_t(
            std::move(out_name), std::move(out_description), std::move(dtype)));

    return *this;
}

const std::vector<op_schema_t::op_parameter_t> &
op_schema_t::get_outputs() const {
    return outputs_;
}

op_schema_t &op_schema_t::set_attr(const std::string &name,
        std::string &&description, bool required, attribute_kind_t attr_kind) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name]
            = attribute_t(name, std::move(description), required, attr_kind);
    return *this;
}

op_schema_t &op_schema_t::set_attr(const std::string &name,
        std::string &&description, bool required, attribute_kind_t attr_kind,
        const char *value) {
    assertm(attributes_.count(name) == 0,
            "provided attribute has already been set");
    attributes_[name] = attribute_t(name, std::move(description), required,
            attr_kind, {std::string(value)});
    return *this;
}

const std::unordered_map<std::string, op_schema_t::attribute_t> &
op_schema_t::get_attrs() const {
    return attributes_;
}

op_schema_t &op_schema_t::set_shape_inference_function(shape_infer_fn fn) {
    tensor_inference_function_ = std::move(fn);
    return *this;
}

shape_infer_fn op_schema_t::get_shape_inference_function() const {
    return tensor_inference_function_;
}

bool op_schema_t::verify_param_num(size_t actual_num,
        const std::set<size_t> &expected_num, param_num_option option) const {
    switch (option) {
        case param_num_option::fixed: {
            // fixed option only has one valid number
            if (expected_num.size() != 1
                    || expected_num.find(actual_num) == expected_num.end()) {
                return false;
            }
        } break;
        case param_num_option::optional: {
            if (expected_num.find(actual_num) == expected_num.end())
                return false;
        } break;
        case param_num_option::variadic: {
            auto lt = expected_num.begin();
            auto gt = expected_num.upper_bound(actual_num);
            auto end = expected_num.end();
            if ((lt != end && *lt > actual_num) || lt == end || gt == end)
                return false;
        } break;
        default: return false;
    }
    return true;
}

bool op_schema_t::verify_param_dtype(
        const std::vector<std::shared_ptr<value_t>> &actual_values,
        const std::vector<op_schema_t::op_parameter_t> &expected_params,
        param_num_option option) const {
    size_t offset = 0;
    for (auto &v : actual_values) {
        const logical_tensor_t &lt = v->get_logical_tensor();
        const std::set<data_type_t> &expected_dtypes
                = expected_params[offset].allowed_dtypes_;
        if (expected_dtypes.find(lt.data_type) == expected_dtypes.end()) {
            return false;
        }
        if (option != param_num_option::variadic) { offset += 1; }
    }

    return true;
}

bool op_schema_t::verify_attributes(
        const std::unordered_map<std::string, utils::attribute_value_t>
                &actual_attrs,
        const std::unordered_map<std::string, attribute_t> &expected_attrs)
        const {
    // check if required attributes are not provided
    for (const auto &elem : expected_attrs) {
        if (elem.second.required_ && actual_attrs.count(elem.first) == 0) {
            return false;
        }
    }
    // check if the data types of actual attributes meet requirements
    for (const auto &elem : actual_attrs) {
        const std::string &attr_name = elem.first;
        if (expected_attrs.count(attr_name) != 0
                && elem.second.get_kind()
                        != expected_attrs.at(attr_name).attr_kind_) {
            return false;
        }
    }

    return true;
}

void op_schema_t::set_default_attribute(op_t *l_op) const {
    const std::unordered_map<std::string, utils::attribute_value_t>
            &actual_attrs = l_op->get_attributes();
    const std::unordered_map<std::string, op_schema_t::attribute_t>
            &expected_attrs = this->get_attrs();
    for (auto iter = expected_attrs.begin(); iter != expected_attrs.end();
            ++iter) {
        // if default attribute not set in op, set it to default value
        if (iter->second.has_default_value_
                && actual_attrs.count(iter->first) == 0) {
            utils::attribute_value_t value = iter->second.attr_;
            const std::string &name = iter->first;
            l_op->set_attr(name, value);
        }
    }
}

bool op_schema_t::verify(const op_t *l_op) const {
    size_t actual_num_inputs = l_op->num_inputs();
    std::set<size_t> expected_num_inputs = get_num_inputs();
    bool param_num_verify_result = verify_param_num(
            actual_num_inputs, expected_num_inputs, inputs_option);
    if (!param_num_verify_result) { return false; }
    bool param_dtype_verify_result = verify_param_dtype(
            l_op->get_input_values(), inputs_, inputs_option);
    if (!param_dtype_verify_result) { return false; }

    size_t actual_num_outputs = l_op->num_outputs();
    std::set<size_t> expected_num_outputs = get_num_outputs();
    param_num_verify_result = verify_param_num(
            actual_num_outputs, expected_num_outputs, outputs_option);
    if (!param_num_verify_result) { return false; }
    param_dtype_verify_result = verify_param_dtype(
            l_op->get_output_values(), outputs_, outputs_option);
    if (!param_dtype_verify_result) { return false; }

    bool attr_verify_result
            = verify_attributes(l_op->get_attributes(), attributes_);
    return attr_verify_result;
}

status_t op_schema_t::shape_infer(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) const {
    shape_infer_fn fn = get_shape_inference_function();
    return fn(n, inputs, outputs);
}

size_t op_schema_t::get_max_valid_param_num(
        const std::set<size_t> &param_num, param_num_option option) const {
    size_t max_valid_num = 0;
    if (option == param_num_option::fixed
            || option == param_num_option::optional) {
        if (!param_num.empty()) { max_valid_num = *param_num.rbegin(); }
    } else {
        max_valid_num = std::numeric_limits<size_t>::max();
    }

    return max_valid_num;
}

void op_schema_t::verify_input_(size_t in_offset) {
    assertm(inputs_offset.find(in_offset) == inputs_offset.end(),
            "provided `in_offset` has already been set");

    inputs_offset.insert(in_offset);
    size_t max_valid_num = get_max_valid_param_num(num_inputs_, inputs_option);
    assertm(max_valid_num > 0, "input set before setting num_inputs_");
    assertm(in_offset < max_valid_num,
            "input offset exceeds declared num of inputs");
    UNUSED(max_valid_num);
}

void op_schema_t::verify_output_(size_t out_offset) {
    assertm(outputs_offset.find(out_offset) == outputs_offset.end(),
            "provided `out_offset` has already been set");

    outputs_offset.insert(out_offset);
    size_t max_valid_num
            = get_max_valid_param_num(num_outputs_, outputs_option);
    assertm(max_valid_num > 0, "output set before setting num_outputs_");
    assertm(out_offset < max_valid_num,
            "output offset exceeds declared num of outputs");
    UNUSED(max_valid_num);
}

op_schema_t &op_schema_t::set_inputs_option(param_num_option option) {
    inputs_option = option;
    return *this;
}

op_schema_t::param_num_option op_schema_t::get_inputs_option() const {
    return inputs_option;
}

op_schema_t &op_schema_t::set_outputs_option(param_num_option option) {
    outputs_option = option;
    return *this;
}

op_schema_t::param_num_option op_schema_t::get_outputs_option() const {
    return outputs_option;
}

op_schema_registry_t::op_schema_registry_once_t::op_schema_registry_once_t(
        op_schema_t &&schema) {
    op_kind_version_schema_map &op_map
            = get_map_without_ensuring_registration();

    const op_kind_t kind = schema.get_op_kind();
    opset_version op_version = schema.get_since_version();

    op_map[kind].insert(std::pair<opset_version, op_schema_t &&>(
            op_version, std::move(schema)));
}

op_kind_version_schema_map &
op_schema_registry_t::get_map_without_ensuring_registration() {
    static op_kind_version_schema_map op_map;
    return op_map;
}

op_kind_version_schema_map &op_schema_registry_t::get_map() {
    op_kind_version_schema_map &op_map
            = get_map_without_ensuring_registration();
    class register_opset_t {
    public:
        register_opset_t() { register_opset_schema(); }
    };
    static register_opset_t ro;

    return op_map;
}

const op_schema_t *op_schema_registry_t::get_op_schema(op_kind_t kind) {
    auto &op_map = get_map();
    if (op_map.count(kind)) {
        return &op_map[kind].rbegin()->second;
    } else {
        return nullptr;
    }
}

void register_schema(op_schema_t &&schema) {
    op_schema_registry_t::op_schema_registry_once_t DNNL_GRAPH_UNUSED
            registration(std::move(schema));
}

} // namespace impl
} // namespace graph
} // namespace dnnl
