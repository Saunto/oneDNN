/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include "softmax.hpp"
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {
namespace ops {

softmax_op::softmax_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    if (outs.empty()) {
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, ins[0]->details_));
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "softmax";
}

std::shared_ptr<sc_graph_t> softmax_op::get_graph_impl() {
    auto graph = std::make_shared<sc_graph_t>();
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    const std::vector<int> &axis = attrs_.get<std::vector<int>>("axis");

    // input
    graph->make_input(inputs);
    // exp(x)
    auto fexp = graph->make("exp", {inputs[0]}, {}, {});

    // sum(exp(x))
    auto freduce = graph->make("reduce", {fexp->get_outputs()[0]}, {},
            {{"need_mean", false}, {"rd_axis", axis}, {"rd_op", 0}});
    // softmax = exp/sum
    auto fdiv = graph->make(
            "div", {fexp->get_outputs()[0], freduce->get_outputs()[0]}, {}, {});
    // output
    graph->make_output(fdiv->get_outputs());
    return graph;
}

void softmax_op::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {}

} // namespace ops

OP_REGISTER(ops::softmax_op, softmax)
} // namespace sc
