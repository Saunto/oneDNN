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
#include "reshape.hpp"
#include <memory>
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {
namespace ops {
static void get_output_shape(sc_dims &outshape, const sc_dims &input_dims,
        const int32_t *shape, int dim, bool special_zero) {
    // we allow one dim value to be -1, which is automatically calculated to
    // keep the number of elements of the out tensor = in tensor.
    int auto_cal_dim = -1;
    size_t total_shape = 1;
    for (int i = 0; i < dim; i++) {
        int shape_v = shape[i];
        if (shape_v == -1) {
            COMPILE_ASSERT(
                    auto_cal_dim == -1, "reshape only support one -1 shape");
            auto_cal_dim = i;
        } else {
            if (special_zero && shape_v == 0) {
                COMPILE_ASSERT(static_cast<size_t>(i) < input_dims.size(),
                        "The special zero at "
                                << i
                                << " dimension is out of range in input shape");
                shape_v = input_dims[i];
            }
            total_shape *= shape_v;
        }
        outshape.emplace_back(shape_v);
    }
    size_t input_total_shape = 1;
    for (auto v : input_dims) {
        input_total_shape *= v;
    }
    const char *error_msg
            = "Reshape: The input tensor size does not match the given shape";
    if (auto_cal_dim != -1) {
        COMPILE_ASSERT(input_total_shape >= total_shape, error_msg);
        outshape[auto_cal_dim] = input_total_shape / total_shape;
    } else {
        COMPILE_ASSERT(input_total_shape == total_shape, error_msg);
    }
}

static_reshape_op::static_reshape_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op("static_reshape_op", ins, outs, attrs) {
    COMPILE_ASSERT(
            attrs.has_key("shape"), "Static reshape requires shape attributes");
    auto shape = attrs.get<sc_dims>("shape");
    std::vector<int32_t> shape_s32(shape.begin(), shape.end());
    bool special_zero = attrs.get<bool>("special_zero");
    auto input_dims = get_inputs()[0]->details_.get_plain_dims();
    auto dim = static_cast<int>(shape.size());
    sc_dims outshape;
    outshape.reserve(dim);
    get_output_shape(outshape, input_dims, shape_s32.data(), dim, special_zero);
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(graph_tensor::make(outshape,
                sc_data_format_t(), get_inputs()[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "Expecting 1 output for reshape");
        auto &details = info_.outputs_[0]->details_;
        COMPILE_ASSERT(details.dtype_ == info_.inputs_[0]->details_.dtype_,
                "Reshape: input/output dtype does not match");
        COMPILE_ASSERT(details.get_plain_dims() == outshape,
                "Reshape: Expecting output shape = "
                        << utils::print_vector(outshape) << ", given: "
                        << utils::print_vector(details.get_plain_dims()));
    }
}
void static_reshape_op::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    throw std::runtime_error("Not implemented");
}

// for single op generate
ir_module_ptr static_reshape_op::get_func(context_ptr ctx) {
    throw std::runtime_error("Not implemented");
}
sc_op_ptr static_reshape_op::constant_optimize(sc_graph_t &graph) {
    auto new_input = graph.make("tensor_view", {get_inputs()[0]}, {},
            {{"shape", get_outputs()[0]->details_.get_plain_dims()}});
    this->replace_uses_with_and_remove(new_input);
    return new_input;
}
} // namespace ops
OP_REGISTER(ops::static_reshape_op, static_reshape);
} // namespace sc
